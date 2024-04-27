'''
GUI for playing around with the system
Features:
- Displays a graph with nodes and arrows (nodes are items, arrows are results)
- Hotkeys:
    - "n" to add nodes
    - "a" to add arrows
    - "x" to delete nodes and arrows
    - "+" to increment n_copies of an arrow you click on
    - "-" to decrement n_copies of an arrow you click on
    - "o" to optimize
    - "d" to drag nodes
- On the bottom, there are buttons for the above, which are green when active
- On the right, the nodes and elo/temperature values are displayed, sorted by elo
- The nodes are draggable
- If arrows exist in both directions, both are displayed but are offset
'''
from elo_optimizer import *
import tkinter as tk
from tkinter import simpledialog
import random

def list_filter(f:Callable, lst:Iterable) -> list:
    '''Same as filter, but returns a list instead of filter object.\n\nShorthand for [x for x in lst if f(x)].'''
    return [x for x in lst if f(x)]

class Node:
    def __init__(self, canvas:tk.Canvas, item:Item|Lottery, 
                 x:float, y:float,
                 size:float=35, cosmetics=None):
        if cosmetics is None:
            rgb = np.random.randint(0, 256, 3)
            # Make it light by shifting s.t. the max is 255
            rgb += (255 - rgb.max())
            cosmetics = {"fill": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", "outline": "black"}
        self.canvas, self.item, self.x, self.y, self.size, self.cosmetics = canvas, item, x, y, size, cosmetics
        self.id = self.canvas.create_oval(x-size, y-size, x+size, y+size, **cosmetics)
        self.text_id = self.canvas.create_text(x, y, 
                                               text=self.generate_text(),
                                               anchor=tk.CENTER, justify=tk.CENTER)
        NODES.append(self)
    def goto(self, x, y):
        self.x, self.y = x, y
        self.canvas.coords(self.id, x-self.size, y-self.size, x+self.size, y+self.size)
        self.canvas.coords(self.text_id, x, y)
        for arrow in ARROWS:
            if self in [arrow.winner_node, arrow.loser_node]:
                arrow.update()
    def generate_text(self, temp_limit=np.inf):
        #return f"{item.name}\n{round(item.elo().item(), 1)} ± {round(item.temperature().item(), 2)}" if item.temperature() < 10 else "??"
        elo_round = lambda x: round(x, 1)
        temp_round = lambda x: round(x, 2 if x < 0.3 else 1)
        return f"{self.item.name}\n{elo_round(self.item.normalized_elo().item())} ± {temp_round(self.item.temperature().item())}" if self.item.temperature() < temp_limit else f"{self.item.name}\n??"
    def delete(self):
        self.item.delete()
        connected = lambda arrow: arrow.winner_node == self or arrow.loser_node == self
        for arrow in filter(connected, ARROWS):
            arrow.delete()
        self.canvas.delete(self.id)
        self.canvas.delete(self.text_id)
        NODES.remove(self) 

class Arrow:
    def offset_to_vector(self):
        x1, y1, x2, y2 = self.winner_node.x, self.winner_node.y, self.loser_node.x, self.loser_node.y
        if x1 == x2 and y1 == y2:
            return np.array([0, 0])
        direction = (x2-x1, y2-y1)
        perpendicular = np.array([-direction[1], direction[0]])
        return self.offset * perpendicular/np.linalg.norm(perpendicular)
    def get_endpoints(self):
        offset_vec = self.offset_to_vector()
        forward_unit = np.array([self.loser_node.x - self.winner_node.x, self.loser_node.y - self.winner_node.y], dtype=float)
        forward_unit /= np.linalg.norm(forward_unit)
        x1 = self.winner_node.x + self.winner_node.size*forward_unit[0] + offset_vec[0]
        y1 = self.winner_node.y + self.winner_node.size*forward_unit[1] + offset_vec[1]
        x2 = self.loser_node.x - self.loser_node.size*forward_unit[0] + offset_vec[0]
        y2 = self.loser_node.y - self.loser_node.size*forward_unit[1] + offset_vec[1]
        return x1, y1, x2, y2
    def update(self):
        x1, y1, x2, y2 = self.get_endpoints()
        self.canvas.coords(self.id, x1, y1, x2, y2)
        self.canvas.coords(self.text_id, (x1+x2)/2, (y1+y2)/2)
        self.canvas.itemconfig(self.text_id, text=str(self.result.n_copies), anchor=tk.CENTER, font=("Arial", 10, "bold") if self.result.n_copies > 1 else ("Arial", 10))
        
    def delete(self):
        self.result.delete()
        ARROWS.remove(self)
        self.canvas.delete(self.id)
        self.canvas.delete(self.text_id)

    def __init__(self, canvas:tk.Canvas, result:Result):
        self.canvas, self.result = canvas, result
        # Find nodes
        winner_nodes = list_filter(lambda node: node.item == self.result.winner, NODES)
        loser_nodes = list_filter(lambda node: node.item == self.result.loser, NODES)
        assert len(winner_nodes) == len(loser_nodes) == 1
        self.winner_node, self.loser_node = winner_nodes[0], loser_nodes[0]
        # Create offset if an arrow with the same nodes but reversed exists
        self.offset = 0
        opposite = lambda arrow: arrow.winner_node == self.loser_node and arrow.loser_node == self.winner_node
        complements = list_filter(opposite, ARROWS)
        assert len(complements) <= 1
        if complements:
            self.offset = 10
            complements[0].offset = 10
        ARROWS.append(self)
        # Canvas stuff
        x1, y1, x2, y2 = self.get_endpoints()
        self.id = canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST)
        self.text_id = canvas.create_text((x1+x2)/2, (y1+y2)/2,
                                          text=str(result.n_copies), anchor=tk.CENTER,
                                          font=("Arial", 10, "bold") if result.n_copies > 1 else ("Arial", 10))

# GLOBALS
NODES:list[Node] = []
ARROWS:list[Arrow] = []

class GUI:
    def set_mode(self, mode:str):
        self.mode, self.cache = mode, []
        for button in self.buttons.values():
            button.config(bg="white")
        self.buttons[mode].config(bg="green")

    def __init__(self, width:int=800, height:int=600):
        self.cache = []
        self.mode = "node"
        self.root = tk.Tk()
        self.root.title("Elo optimizer")
        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        self.buttons = {}
        mode_dicts =   {"node": {"hotkey": "n", "description": "Add node"},
                        "arrow": {"hotkey": "a", "description": "Add arrow"},
                        "delete": {"hotkey": "x", "description": "Delete nodes/arrows"},
                        "increment": {"hotkey": "+", "description": "Increment n_copies"},
                        "decrement": {"hotkey": "-", "description": "Decrement n_copies"},
                        "drag": {"hotkey": "d", "description": "Drag nodes"},
                        "optimize": {"hotkey": "o", "description": "Optimize", "command": lambda event=None: self.optimize()},
                        "optimize_from_scratch": {"hotkey": "O", "description": "Optimize from scratch", "command": lambda event=None: self.optimize(reinit=True)},
                        "lottery": {"hotkey": "l", "description": "Add lottery"}
                        }
        for mode, mode_dict in mode_dicts.items():
            command = (lambda event=None, mode=mode: self.set_mode(mode)) if "command" not in mode_dict else mode_dict["command"]
            self.buttons[mode] = tk.Button(self.root, 
                                           text=f"{mode_dict["description"]} ({mode_dict["hotkey"]})", 
                                           command=command)
            self.buttons[mode].pack(side=tk.LEFT)
            # Bind hotkey
            self.root.bind(f"<Key-{mode_dict["hotkey"]}>", command)
        self.set_mode("node")
        self.root.bind("<Button-1>", self.click)
        self.root.bind("<B1-Motion>", self.drag)

    def str_to_rgb(self, string:str):
        scaled = self.canvas.winfo_rgb(string)
        return np.array([scaled[0]//256, scaled[1]//256, scaled[2]//256])

    def click(self, event):
        # If a button is pressed, do nothing
        if event.widget != self.canvas:
            return
        # Node mode
        if self.mode == "node":
            # Name based on number of nodes
            name = chr(65 + len(NODES))
            median_elo = torch.median(stack([item.elo() for item in ITEMS])) if ITEMS else 0
            item = Item(name, median_elo, 1)
            Node(self.canvas, item, event.x, event.y)
        # Arrows and lotteries (similar logic because they both involve selecting two nodes)
        if self.mode in ["arrow", "lottery"]:
            clicked = lambda node: np.linalg.norm(np.array([node.x, node.y]) - np.array([event.x, event.y])) < node.size
            nodes = list_filter(clicked, NODES)
            if nodes:
                self.cache.append(nodes[0])
                if len(self.cache) == 2:
                    first, second = self.cache
                    if first != second:
                        if self.mode == "arrow":
                            # First item is the winner
                            result = add_result(first.item, second.item)
                            existing_arrows = list_filter(lambda arrow: arrow.result == result, ARROWS)
                            if existing_arrows:
                                existing_arrows[0].update()
                            else:
                                Arrow(self.canvas, result)
                        elif self.mode == "lottery" and isinstance(first.item, Item) and isinstance(second.item, Item):
                            # Pop up a window to input the weight of the first item
                            weight = simpledialog.askfloat("Lottery", f"Enter the weight of {first.item.name} (0-1)")
                            if weight is not None:
                                w1, w2 = weight, 1-weight
                                lottery = Lottery([first.item, second.item], [w1, w2])
                                # Add a node for the lottery
                                # Average the colors and positions
                                x = w1*first.x + w2*second.x
                                y = w1*first.y + w2*second.y
                                # Reconstruct colors from strings
                                c1, c2 = [self.str_to_rgb(node.cosmetics["fill"]) for node in [first, second]]
                                color = (w1*c1 + w2*c2).round().astype(int)
                                cosmetics = {"fill": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", "outline": "black"}
                                Node(self.canvas, lottery, x, y, cosmetics=cosmetics)
                    self.cache = []
        # Delete mode
        if self.mode == "delete":
            # Use canvas.find_withtag(tk.CURRENT) to find items
            objects = self.canvas.find_withtag(tk.CURRENT)
            to_delete = set(filter(lambda node: node.id in objects, NODES + ARROWS))
            for obj in to_delete:
                obj.delete()
            
        # Increment/decrement mode
        if self.mode in ["increment", "decrement"]:
            # Use canvas.find_withtag(tk.CURRENT) to find the arrow
            arrows = self.canvas.find_withtag(tk.CURRENT)
            if arrows:
                arrow = list_filter(lambda arrow: arrow.id == arrows[0], ARROWS)[0]
                arrow.result.n_copies += 1 if self.mode == "increment" else -1
                if arrow.result.n_copies == 0:
                    arrow.delete()
                else:
                    arrow.update()
    
    def drag(self, event):
        if self.mode == "drag":
            clicked = lambda node: np.linalg.norm(np.array([node.x, node.y]) - np.array([event.x, event.y])) < node.size
            nodes = list_filter(clicked, NODES)
            if nodes:
                node = nodes[0]
                node.goto(event.x, event.y) # Also updates arrows

    def optimize(self, reinit=False):
        if not (RESULTS and PARAMETERS):
            return
        if reinit:
            for item in ITEMS:
                item.store(random.uniform(-10, 10), random.uniform(1, 10))
        full_batch_optimize(1000, 0.1)
        for node in NODES:
            self.canvas.itemconfig(node.text_id, 
                                   text=node.generate_text())

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = GUI(width=1000, height=600)
    gui.run()