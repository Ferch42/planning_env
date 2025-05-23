from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Collection, Set, Tuple, DefaultDict
from collections import deque, defaultdict
import random
import numpy as np
from tqdm import tqdm

@dataclass(frozen=True)
class State:
    holding: FrozenSet[str]
    available: FrozenSet[str]

class CraftingDomain:
    #VALID_ACTIONS = {'hold', 'drop', 'build', 'decompose'}
    VALID_ACTIONS = {'hold', 'drop', 'build'}
    BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
    
    def __init__(self, allowed_actions: Collection[str] = VALID_ACTIONS,
                 allowed_items: Collection[str] = None):
        self.RECIPES: Dict[str, FrozenSet[str]] = {
            'Basic_Engine': frozenset({'Iron', 'Fuel'}),
            'Thermal_Core': frozenset({'Copper', 'Stone'}),
            'Hybrid_Drive': frozenset({'Basic_Engine', 'Thermal_Core'}),
            'Aerial_Transport': frozenset({'Hybrid_Drive', 'Wood'}),
            'Reinforced_Frame': frozenset({'Basic_Engine', 'Wood'}),
            'Steam_Generator': frozenset({'Iron', 'Fuel', 'Stone'}),
            'Steam_Cart': frozenset({'Steam_Generator', 'Wood'}),
            'Copper_Furnace': frozenset({'Fuel', 'Copper', 'Wood'})
        }

        self.CRAFTED_ITEMS = set(self.RECIPES.keys())
        self.MAX_CAPACITY = 3
        self.allowed_actions = set(allowed_actions)
        self.allowed_items = set(allowed_items) if allowed_items else self.BASIC_ITEMS | self.CRAFTED_ITEMS

    def hold(self, state: State, item: str) -> Optional[State]:
        if (item in self.allowed_items and
            len(state.holding) < self.MAX_CAPACITY and 
            item in state.available):
            new_holding = set(state.holding)
            new_holding.add(item)
            new_available = set(state.available)
            new_available.remove(item)
            return State(frozenset(new_holding), frozenset(new_available))
        return None

    def drop(self, state: State, item: str) -> Optional[State]:
        if item in state.holding:
            new_holding = set(state.holding)
            new_holding.remove(item)
            new_available = set(state.available)
            new_available.add(item)
            return State(frozenset(new_holding), frozenset(new_available))
        return None

    def build(self, state: State, target: str) -> Optional[State]:
        components = self.RECIPES.get(target)
        if components and components.issubset(state.holding):
            new_holding = set(state.holding) - components
            new_holding.add(target)
            return State(frozenset(new_holding), state.available)
        return None

    def decompose(self, state: State, item: str) -> Optional[State]:
        components = self.RECIPES.get(item)
        if components and item in state.holding and (len(state.holding) - 1 + len(components) <= self.MAX_CAPACITY):
            new_holding = set(state.holding)
            new_holding.remove(item)
            new_holding.update(components)
            return State(frozenset(new_holding), state.available)
        return None

    def find_plan(self, goal: str, available_basics: Set[str], inventory: Set[str]) -> List[str]:
        initial_available = (self.BASIC_ITEMS & available_basics) | set()
        initial_state = State(frozenset(inventory), frozenset(initial_available))
        queue = deque([(initial_state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()
            
            if goal in current_state.holding:
                return path
            
            state_sig = (current_state.holding, current_state.available)
            if state_sig in visited:
                continue
            visited.add(state_sig)

            if 'hold' in self.allowed_actions:
                for item in current_state.available:
                    if new_state := self.hold(current_state, item):
                        queue.append((new_state, path + [f"hold {item}"]))
            
            if 'drop' in self.allowed_actions:
                for item in current_state.holding:
                    if new_state := self.drop(current_state, item):
                        queue.append((new_state, path + [f"drop {item}"]))
            
            if 'build' in self.allowed_actions:
                for target in self.RECIPES:
                    if new_state := self.build(current_state, target):
                        queue.append((new_state, path + [f"build {target}"]))
            
            if 'decompose' in self.allowed_actions:
                for item in current_state.holding:
                    if new_state := self.decompose(current_state, item):
                        queue.append((new_state, path + [f"decompose {item}"]))
        
        return []

class GridWorld:
    BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
    
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.item_locations = defaultdict(set)
        self.reset_world()

    def reset_world(self):
        # Clear existing items
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = None
        self.item_locations.clear()

        # Place 2 of each basic item at random positions
        for item in self.BASIC_ITEMS:
            placed = 0
            while placed < 10:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                if self.grid[y][x] is None:
                    self.grid[y][x] = item
                    self.item_locations[item].add((x, y))
                    placed += 1

    def remove_item(self, x: int, y: int) -> Optional[str]:
        item = self.grid[y][x]
        if item:
            self.grid[y][x] = None
            self.item_locations[item].discard((x, y))
            
            # Respawn basic items to maintain count
            if item in self.BASIC_ITEMS:
                # Find empty cells
                empty_cells = [(x, y) for y in range(self.height) 
                             for x in range(self.width) if self.grid[y][x] is None]
                if empty_cells:
                    new_x, new_y = random.choice(empty_cells)
                    self.place_item(new_x, new_y, item)
                    
        return item

    def place_item(self, x: int, y: int, item: str) -> bool:
        if self.grid[y][x] is None:
            self.grid[y][x] = item
            self.item_locations[item].add((x, y))
            return True
        return False

class Agent:
    def __init__(self, x=25, y=25):
        self.x = x
        self.y = y
        self.inventory = []
        self.max_inventory = 3
        self.discovered_resources = defaultdict(set)
        self.visitation_counts = defaultdict(int)
        # Initialize starting position count
        self.visitation_counts[(x, y)] = 1

    def move(self, dx: int, dy: int, grid: GridWorld) -> bool:
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
            # Update position and visitation count
            self.x, self.y = new_x, new_y
            self.visitation_counts[(new_x, new_y)] += 1
            return True
        return False

    def collect(self, grid: GridWorld) -> Tuple[bool, str]:
        if len(self.inventory) >= self.max_inventory:
            return False, "Inventory full"
        item = grid.remove_item(self.x, self.y)
        if item:
            self.inventory.append(item)
            # Remove collected location from discovered resources
            self.discovered_resources[item].discard((self.x, self.y))
            return True, f"Collected {item}"
        return False, "No items here"

    def drop(self, grid: GridWorld, item: str) -> Tuple[bool, str]:
        if item not in self.inventory:
            return False, "Item not in inventory"
        
        # Keep trying until placement succeeds
        while True:
            # Try to place the item in the current position
            if grid.place_item(self.x, self.y, item):
                self.inventory.remove(item)
                self.discovered_resources[item].add((self.x, self.y))
                return True, f"Dropped {item}"
            
            # If current cell is occupied, move to a random adjacent cell
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)  # Randomize direction order
            
            moved = False
            for dx, dy in directions:
                # Attempt to move in this direction
                if self.move(dx, dy, grid):
                    moved = True
                    break  # Move once and retry
            
            if not moved:
                # Edge case: agent is at grid boundary and all moves are invalid
                # This should be extremely rare given grid size, but continue trying
                continue

        return False
    
    def destroy(self, item: str) -> Tuple[bool, str]:
        """Destroy an item from inventory without placing it in the world"""
        if item not in self.inventory:
            return False, "Item not in inventory"
        self.inventory.remove(item)
        return True, f"Destroyed {item}"

    
    def choose_direction(self, grid: GridWorld) -> Tuple[int, int]:
        """Choose direction based on least-visited adjacent cells"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbor_counts = []
        
        # Evaluate all possible moves
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                count = self.visitation_counts.get((nx, ny), 0)
                neighbor_counts.append((count, dx, dy))
        
        # Find minimum visitation count
        if not neighbor_counts:
            return (0, 0)  # Can't move
        min_count = min([c[0] for c in neighbor_counts])
        candidates = [(dx, dy) for cnt, dx, dy in neighbor_counts if cnt == min_count]
        return random.choice(candidates)

class Game:
    def __init__(self):
        self.grid = GridWorld()
        self.agent = Agent()
        self.crafting_domain = CraftingDomain()

    def automated_crafting_mission(self, list_of_targets: List[str], max_steps=1_000_000):
        #print(f"\n🚀 Starting mission to craft {list_of_targets}!")
        
        timestep_list = []
        t = 0
        steps = 0
        
        target = list_of_targets.pop(0)
        while steps < max_steps:
            
            #if steps % 10000 == 0:
            #    self.print_status(steps)
            
            if target in self.agent.inventory:

                self.agent.destroy(target)
                timestep_list.append(steps-t)
                t = steps
                #print(f'Sucessfully built {target} in {steps}')
                #self.print_status(steps)
                
                try:
                    target = list_of_targets.pop(0)
                except:
                    break
                

            self.update_discoveries()
            known_basics = {item for item in self.grid.BASIC_ITEMS 
                           if self.agent.discovered_resources[item]}
            
            plan = self.crafting_domain.find_plan(target, known_basics, set(self.agent.inventory))
            
            if not plan:
                self.systematic_exploration()
                steps += 1
                continue
            
            for action in plan:
                if steps >= max_steps:
                    break
                try:
                    steps_taken = self.execute_action(action)
                    steps += steps_taken
                    self.update_discoveries()
                except Exception as e:
                    print(f"Action failed: {action} - {str(e)}")
                    break
        
        return timestep_list
        

    def execute_action(self, action: str) -> int:
        cmd, *rest = action.split()
        item = ' '.join(rest)
        
        if cmd == "hold":
            return self.collect_item(item)
        elif cmd == "drop":
            success, msg = self.agent.drop(self.grid, item)
            if not success:
                raise RuntimeError(msg)
            return 1
        elif cmd == "build":
            self.craft_item(item)
            return 1
        elif cmd == "decompose":
            self.decompose_item(item)
            return 1
        else:
            raise ValueError(f"Unknown action: {action}")

    def collect_item(self, item: str) -> int:
        if not self.agent.discovered_resources[item]:
            raise ValueError(f"No known {item} locations")
        
        original_x = self.agent.x
        original_y = self.agent.y
        
        closest = min(self.agent.discovered_resources[item],
                     key=lambda p: abs(p[0]-original_x) + abs(p[1]-original_y))
        
        # Calculate Manhattan distance
        distance = abs(closest[0] - original_x) + abs(closest[1] - original_y)
        
        # Update agent position (simulate movement)
        self.agent.x, self.agent.y = closest
        
        # Perform collection
        success, msg = self.agent.collect(self.grid)
        if not success:
            raise RuntimeError(msg)
        
        # Return total steps taken: distance + 1 (collect action)
        return distance + 1

    def craft_item(self, target: str):
        components = self.crafting_domain.RECIPES.get(target)
        if not components:
            raise ValueError(f"Unknown recipe: {target}")
        
        for c in components:
            if c not in self.agent.inventory:
                raise RuntimeError(f"Missing component: {c}")
        
        for c in components:
            self.agent.inventory.remove(c)
        self.agent.inventory.append(target)

    def decompose_item(self, item: str):
        components = self.crafting_domain.RECIPES.get(item)
        if not components:
            raise ValueError(f"Cannot decompose {item}")
        
        if item not in self.agent.inventory:
            raise RuntimeError(f"Not carrying {item}")
        
        if len(self.agent.inventory) - 1 + len(components) > self.agent.max_inventory:
            raise RuntimeError("Inventory full")
        
        self.agent.inventory.remove(item)
        self.agent.inventory.extend(components)

    def update_discoveries(self):
        for item in self.grid.item_locations:
            for x, y in self.grid.item_locations[item]:
                if abs(x-self.agent.x) <= 5 and abs(y-self.agent.y) <= 5:
                    self.agent.discovered_resources[item].add((x, y))

    def random_exploration(self):
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.agent.move(dx, dy, self.grid)

    def systematic_exploration(self):
        """Replaces random_exploration with systematic movement"""
        dx, dy = self.agent.choose_direction(self.grid)
        self.agent.move(dx, dy, self.grid)

    def print_status(self, steps: int):
        print(f"\n⏱ Step {steps} Report:")
        print(f"📍 Position: ({self.agent.x}, {self.agent.y})")
        print(f"🎒 Inventory ({len(self.agent.inventory)}/{self.agent.max_inventory}): {self.agent.inventory}")
        print("🔍 Discovered Resources:")
        for item, locs in self.agent.discovered_resources.items():
            print(f"  - {item}: {len(locs)} locations")

    def mission_result(self, target: str, steps: int):
        if target in self.agent.inventory:
            print(f"\n🎉 Successfully crafted {target} in {steps} steps!")
            print(f"📍 Final position: ({self.agent.x}, {self.agent.y})")
            print(f"🎒 Final inventory: {self.agent.inventory}")
        else:
            print("\n🔥 Mission failed: Timeout reached")

if __name__ == "__main__":
    game = Game()
    BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
    basic_itemlist = random.choices(list(BASIC_ITEMS), k=20)
    basic_itemlist = ['Iron']*20

    answ = []
    for i in tqdm(range(10_000)):
        game = Game()
        answ.append(game.automated_crafting_mission(basic_itemlist.copy()))
        #print(len(answ[-1]))
        if i%1_000==0:

            print("Basic Items")
            print(np.array(answ).shape)
            print(list(np.mean(np.array(answ), axis = 0)))

    print("Basic Items")
    print(np.array(answ).shape)
    print(list(np.mean(np.array(answ), axis = 0)))
    """
    answ = []
    for i in tqdm(range(100)):
        answ.append(game.automated_crafting_mission(["Hybrid_Drive"]*50))
        #print(len(answ[-1]))

    print("Hybrid_Drive")
    print(np.array(answ).shape)
    print(list(np.mean(np.array(answ), axis = 0)))
    
    BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
    basic_itemlist = random.choices(list(BASIC_ITEMS), k=20)

    answ = []
    for i in tqdm(range(1000)):
        answ.append(game.automated_crafting_mission(basic_itemlist.copy()))
        #print(len(answ[-1]))

    print("Basic Items")
    print(np.array(answ).shape)
    print(list(np.mean(np.array(answ), axis = 0)))

    COMPLEX_ITEMS_1 = {'Basic_Engine', 'Thermal_Core', 'Steam_Generator', 'Copper_Furnace' }
    complex_itemlist_1 = random.choices(list(COMPLEX_ITEMS_1), k=50)

    answ = []
    for i in tqdm(range(100)):
        answ.append(game.automated_crafting_mission(complex_itemlist_1.copy()))
        #print(len(answ[-1]))

    print("Complex Items 1")
    print(np.array(answ).shape)
    print(list(np.mean(np.array(answ), axis = 0)))


    COMPLEX_ITEMS_2 = {'Aerial_Transport', 'Reinforced_Frame', 'Steam_Cart' }

    complex_itemlist_2 = random.choices(list(COMPLEX_ITEMS_2), k=50)

    answ = []
    for i in tqdm(range(100)):
        answ.append(game.automated_crafting_mission(complex_itemlist_2.copy()))
        #print(len(answ[-1]))

    print("Complex Items 2")
    print(np.array(answ).shape)
    print(list(np.mean(np.array(answ), axis = 0)))
    
    """