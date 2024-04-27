'''
To use elo_optimizer.py from your own file:
- Import it. It is a stateful module with globals:
      ITEMS: list[Item]  
      LOTTERIES: list[Lottery]  
      RESULTS: list[Result]  
      PARAMETERS: list[torch.Tensor]  
      MODE = "compound" # "compound" or "overwrite" (mode for adding results for pairs that have already been played)  
- Interact with it as follows:  
    - Use the Item() and Lottery() constructors to create items and lotteries  
    - Use add_result() to add results (don't use the Result constuctor directly)  
    - Items, Lotteries, and Results have delete() methods. The delete() method removes an object, its dependents, and its parameters from global lists.
    - Call optimization methods as needed to recompute the best fit utilities and temperatures. Currently only full_batch_optimize() exists.  
    - Interact directly with the global lists for other operations  
'''
import numpy as np
import torch
from torch import stack, logsumexp
from torch.nn.functional import log_softmax
from dataclasses import dataclass
from typing import Iterable, Callable

def list_filter(f:Callable, lst:Iterable) -> list:
    '''Same as filter, but returns a list instead of filter object.\n\nShorthand for [x for x in lst if f(x)].'''
    return [x for x in lst if f(x)]

class Item:
    def store(self, elo, temperature):
        elo = float(elo)
        # Remove old parameters from PARAMETERS
        PARAMETERS[:] = [p for p in PARAMETERS if p not in self.parameters()]
        # Store new parameters
        self.params["elo"] = torch.tensor(elo, requires_grad=True)
        self.params["log_temp"] = torch.tensor(np.log(temperature), requires_grad=True)
        PARAMETERS.extend(self.parameters())

    def elo(self):
        return self.params["elo"]
    def temperature(self):
        return torch.exp(self.params["log_temp"])
    def parameters(self):
        assert all(param.requires_grad for param in self.params.values())
        return self.params.values()
    
    def __init__(self, name:str, elo, temperature):
        self.name = name
        self.params:dict[str, torch.Tensor] = {}
        self.store(elo, temperature) # This also adds the parameters to PARAMETERS
        ITEMS.append(self)

    def delete(self):
        PARAMETERS[:] = [p for p in PARAMETERS if p not in self.parameters()]
        RESULTS[:] = [r for r in RESULTS if r.winner != self and r.loser != self]
        for l in LOTTERIES:
            if self in l.items:
                l.delete()
        ITEMS.remove(self)
        
    def normalized_elo(self):
        median = torch.median(stack([item.elo() for item in ITEMS]))
        return self.elo() - median
    
class Lottery:
    def elo(self):
        return torch.sum(self.weights * stack([item.elo() for item in self.items]))
    def temperature(self):
        # Multiply temperatures by weights then logsumexp
        temps = stack([item.temperature() for item in self.items])
        return logsumexp(self.weights * temps, dim=0)
    
    def __init__(self, items:Iterable[Item], weights, name_joiner:str="+"):
        assert len(items) == len(weights) > 0 and all(isinstance(item, Item) for item in items)
        name_array = [f"{weight}*{item.name}" for item, weight in zip(items, weights)]
        self.name = name_joiner.join(name_array)
        self.items = items
        # Normalize weights to sum to 1
        self.weights = torch.tensor(weights, requires_grad=False) / sum(weights)
        LOTTERIES.append(self)

    def delete(self):
        RESULTS[:] = [r for r in RESULTS if r.winner != self and r.loser != self]
        LOTTERIES[:] = [l for l in LOTTERIES if l != self]
    def normalized_elo(self):
        median = torch.median(stack([item.elo() for item in ITEMS]))
        return self.elo() - median

@dataclass
# Note: Always create a new result with add_result (assuming you want it to be stored to RESULTS)
class Result:
    winner: Item | Lottery
    loser: Item | Lottery
    n_copies: int = 1
    def get_logP(self):
        return logP(self.winner, self.loser) * self.n_copies
    def delete(self):
        RESULTS[:] = [r for r in RESULTS if r != self]

# GLOBALS
ITEMS:list[Item] = []
LOTTERIES:list[Lottery] = [] # WARNING: Currently unused and untested
RESULTS:list[Result] = []
PARAMETERS:list[torch.Tensor] = []
MODE = "compound" # "compound" or "overwrite" (mode for adding results for pairs that have already been played)

# Recommended method for adding results. Also returns the result object.
def add_result(winner:Item, loser:Item):
    if MODE == "compound":
        for r in RESULTS:
            if r.winner == winner and r.loser == loser:
                r.n_copies += 1
                return r
        RESULTS.append(Result(winner, loser))
    elif MODE == "overwrite":
        different_pair = lambda x: {x.winner, x.loser} != {winner, loser}
        RESULTS[:] = list_filter(different_pair, RESULTS) + [Result(winner, loser)]
    return RESULTS[-1]

def logP(winner, loser):
    temps = stack([winner.temperature(), loser.temperature()])
    joint_temp = logsumexp(temps, dim=0) # This is a hacky approximation
    logProb = log_softmax(stack([winner.elo()/joint_temp, loser.elo()/joint_temp]), dim=0)[0]
    assert not torch.isnan(logProb) and not torch.isinf(logProb) and 0 > logProb > -np.inf and logProb.requires_grad
    return logProb

def loss(results):
    return -sum((result.get_logP() for result in results))

# Uses full batch Adam to optimize all elos and temperatures
def full_batch_optimize(steps:int, lr:float,
                        min_temperature:float=1, max_temperature:float=10):
    print_rule = lambda i: i < 10 or i % (steps//10) == 0 or i == steps-1
    opt = torch.optim.Adam(PARAMETERS, lr=lr)
    for i in range(steps):
        opt.zero_grad()
        l = loss(RESULTS)
        l.backward()
        # Print stuff
        if print_rule(i):
            grad_norm = torch.sqrt(sum(torch.norm(param.grad)**2 for param in PARAMETERS))
            print(f"Step {i}, loss {l}, grad_norm {grad_norm}")
        # Take a step
        opt.step()
        # Clamp temperatures
        for item in ITEMS:
            item.params["log_temp"].data.clamp_(np.log(min_temperature), np.log(max_temperature))

'''
Notes on temperature behavior:
Currently, in some circular cases, it just "gives up" on some items and sets their 
temperature to a very high value.
Some settings which could change this behavior:
- Change the loss function to penalize high temperatures
- Restrict the temperature to a certain range

Notes on efficiency:
With large numbers of items, the current implementation is not efficient.
Things that could be done:
- Use mini-batches
- Only optimize after every N matches
- Only optimize the items involved in the last N matches (and maybe also their neighbors)
    - In this case, filter RESULTS before calling loss

'''

'''
Tests
'''
def test_logP():
    print(torch.exp(logP(Item('a', 0, 1), Item('b', 0, 1)))) # 0.5
    print(torch.exp(logP(Item('a', 0, 1), Item('b', 0, 2)))) # 0.5
    print(torch.exp(logP(Item('a', 2, 1), Item('b', 0, 1)))) # Higher than 0.5

def load_preset(preset:str, *items:Item):
    if preset == "circular + A>C":
        a, b, c, d, e, f, g = items
        add_result(a, b)
        add_result(b, c)
        add_result(c, d)
        add_result(d, e)
        add_result(e, f)
        add_result(f, g)
        add_result(g, a)
        add_result(a, c)
    elif preset == "scratch":
        a, b, c, d, e, f, g = items
        # a > b > c > d > e > f > g
        # Correct ones
        add_result(a, b)
        add_result(b, c)
        add_result(c, d)
        add_result(d, e)
        add_result(e, f)
        add_result(f, g)
        add_result(a, c)
        add_result(a, d)
        add_result(b, d)
        add_result(b, f)
        add_result(c, e)
        # Upsets
        add_result(b, a)
        add_result(e, b)
        add_result(g, c)
    else:
        raise ValueError(f"Preset {preset} not found")

def test_full_batch_optimize():
    items = [Item(name, 0, 1) for name in "abcdefg"]
    a, b, c, d, e, f, g = items
    preset_name = "scratch" #"circular + A>C"
    load_preset(preset_name, a, b, c, d, e, f, g)
    full_batch_optimize(1000, 0.1)
    def round_(x): return round(x.item(), 2)
    #print([round_(item.elo()) for item in items])
    print([round_(item.elo()) if item.temperature() < 10 else "??" for item in items])
    print([round_(item.temperature()) for item in items])

if __name__ == "__main__":
    test_full_batch_optimize()
    pass
    