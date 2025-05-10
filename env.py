from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Collection, Set, Tuple, DefaultDict
from collections import deque, defaultdict
import random

@dataclass(frozen=True)
class State:
    holding: FrozenSet[str]
    available: FrozenSet[str]

class CraftingDomain:
    VALID_ACTIONS = {'hold', 'drop', 'build', 'decompose'}
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

    def find_plan(self, goal: str, available_basics: Set[str]) -> List[str]:
        initial_available = (self.BASIC_ITEMS & available_basics) | set()
        initial_state = State(frozenset(), frozenset(initial_available))
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
            while placed < 2:
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
        self.visited_cells = set()

    def move(self, dx: int, dy: int, grid: GridWorld) -> bool:
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
            self.x, self.y = new_x, new_y
            self.visited_cells.add((new_x, new_y))
            return True
        return False

    def collect(self, grid: GridWorld) -> Tuple[bool, str]:
        if len(self.inventory) >= self.max_inventory:
            return False, "Inventory full"
        item = grid.remove_item(self.x, self.y)
        if item:
            self.inventory.append(item)
            return True, f"Collected {item}"
        return False, "No items here"

    def drop(self, grid: GridWorld, item: str) -> Tuple[bool, str]:
        if item not in self.inventory:
            return False, "Item not in inventory"
        if grid.place_item(self.x, self.y, item):
            self.inventory.remove(item)
            self.discovered_resources[item].add((self.x, self.y))
            return True, f"Dropped {item}"
        return False, "Cell occupied"

class Game:
    def __init__(self):
        self.grid = GridWorld()
        self.agent = Agent()
        self.crafting_domain = CraftingDomain()

    def automated_crafting_mission(self, target: str, max_steps=10000):
        print(f"\n🚀 Starting mission to craft {target}!")
        steps = 0
        
        while steps < max_steps and target not in self.agent.inventory:
            if steps % 100 == 0:
                self.print_status(steps)
            
            self.update_discoveries()
            known_basics = {item for item in self.grid.BASIC_ITEMS 
                           if self.agent.discovered_resources[item]}
            
            plan = self.crafting_domain.find_plan(target, known_basics)
            
            if not plan:
                self.random_exploration()
                steps += 1
                continue
            
            for action in plan:
                if steps >= max_steps:
                    break
                try:
                    self.execute_action(action)
                except Exception as e:
                    print(f"Action failed: {action} - {str(e)}")
                    break
                steps += 1
                self.update_discoveries()

        self.mission_result(target, steps)

    def execute_action(self, action: str):
        cmd, *rest = action.split()
        item = ' '.join(rest)
        
        if cmd == "hold":
            self.collect_item(item)
        elif cmd == "drop":
            success, msg = self.agent.drop(self.grid, item)
            if not success:
                raise RuntimeError(msg)
        elif cmd == "build":
            self.craft_item(item)
        elif cmd == "decompose":
            self.decompose_item(item)
        else:
            raise ValueError(f"Unknown action: {action}")

    def collect_item(self, item: str):
        if not self.agent.discovered_resources[item]:
            raise ValueError(f"No known {item} locations")
        
        closest = min(self.agent.discovered_resources[item],
                     key=lambda p: abs(p[0]-self.agent.x) + abs(p[1]-self.agent.y))
        self.agent.x, self.agent.y = closest
        success, msg = self.agent.collect(self.grid)
        if not success:
            raise RuntimeError(msg)

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
    game.automated_crafting_mission("Iron")