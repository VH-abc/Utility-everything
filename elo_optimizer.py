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
from numpy import log
import torch
from torch import stack, logsumexp, tensor
from torch.nn.functional import log_softmax
from dataclasses import dataclass
from typing import Iterable, Callable

def purify(lst, condition:Callable):
    lst[:] = [*filter(condition, lst)]

class Deletable:
    def delete(self):
        for lst in [ITEMS, LOTTERIES, RESULTS]:
            purify(lst, lambda x: x != self)
        purify(RESULTS, lambda r: self not in [r.winner, r.loser])

class IL_base(Deletable):
    def normalized_elo(self):
        median = torch.median(stack([item.elo() for item in ITEMS]))
        return self.elo() - median

class Item(IL_base):
    def store(self, elo, temperature):
        # Remove old parameters from PARAMETERS
        purify(PARAMETERS, lambda p: p not in self.parameters())
        # Store new parameters
        self.params["elo"] = tensor(float(elo), requires_grad=True)
        self.params["log_temp"] = tensor(log(temperature), requires_grad=True)
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
        self.params = {}
        self.store(elo, temperature) # This also adds the parameters to PARAMETERS
        ITEMS.append(self)

    def delete(self):
        super().delete()
        purify(PARAMETERS, lambda p: p not in self.parameters())
        for l in LOTTERIES:
            if self in l.items:
                l.delete()
    
class Lottery(IL_base):
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
        self.weights = tensor(weights, requires_grad=False) / sum(weights)
        LOTTERIES.append(self)

@dataclass
# Note: Always create a new result with add_result (assuming you want it to be stored to RESULTS)
class Result(Deletable):
    winner: Item | Lottery
    loser: Item | Lottery
    n_copies: int = 1
    def get_logP(self):
        return logP(self.winner, self.loser) * self.n_copies

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
    elif MODE == "overwrite":
        different_pair = lambda x: not ((x.winner == winner and x.loser == loser) or (x.winner == loser and x.loser == winner))
        purify(RESULTS, different_pair)
    RESULTS.append(Result(winner, loser))
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
        (l := loss(RESULTS)).backward()
        # Print stuff
        if print_rule(i):
            print(f"Step {i}, loss {l}, grad_norm {torch.sqrt(sum(torch.norm(param.grad)**2 for param in PARAMETERS))}")
        # Take a step
        opt.step()
        # Clamp temperatures
        for item in ITEMS:
            item.params["log_temp"].data.clamp_(log(min_temperature), log(max_temperature))
            

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
TESTS 
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
    