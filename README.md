So far, only the core utility logic and back-end GUI are implemented. The EloEverything-style ranking interface isn't implemented yet.  
Run gui.py to play with the algorithm.

<img width="615" alt="image" src="https://github.com/VH-abc/Utility-everything/assets/76539808/10227d68-aa1c-4e70-b6c8-3f7c0b827d2f">
<br><br>

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
