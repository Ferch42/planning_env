import random
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional

class GridWorld:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.initial_resources = {
            'Iron': [(5, 5), (45, 45)],
            'Fuel': [(5, 45), (45, 5)],
            'Copper': [(15, 10), (35, 40)],
            'Stone': [(10, 25), (40, 25)],
            'Wood': [(20, 20), (30, 30)]
        }
        self.item_locations = defaultdict(set)
        self.reset_world()
    
    def reset_world(self):
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.item_locations.clear()
        for item, positions in self.initial_resources.items():
            for x, y in positions:
                self.grid[y][x] = item
                self.item_locations[item].add((x, y))
    
    def get_cell_items(self, x: int, y: int) -> List[str]:
        return [self.grid[y][x]] if self.grid[y][x] else []
    
    def remove_item(self, x: int, y: int) -> Optional[str]:
        if self.grid[y][x]:
            item = self.grid[y][x]
            self.grid[y][x] = None
            self.item_locations[item].discard((x, y))
            return item
        return None
    
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
        self.steps = 0
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
    
    def collect_item(self, grid: GridWorld) -> Tuple[bool, str]:
        if len(self.inventory) >= self.max_inventory:
            return False, "Backpack full!"
        
        item = grid.remove_item(self.x, self.y)
        if item:
            if (self.x, self.y) in self.discovered_resources[item]:
                if not grid.get_cell_items(self.x, self.y):
                    self.discovered_resources[item].discard((self.x, self.y))
            self.inventory.append(item)
            return True, f"Collected {item}"
        return False, "No items here!"
    
    def drop_item(self, grid: GridWorld, item: str) -> Tuple[bool, str]:
        if item not in self.inventory:
            return False, "Item not in inventory"
        if grid.place_item(self.x, self.y, item):
            self.record_discovery(item, self.x, self.y)
            self.inventory.remove(item)
            return True, f"Placed {item}"
        return False, "Cell occupied"
    
    def record_discovery(self, item: str, x: int, y: int):
        self.discovered_resources[item].add((x, y))

class RecipeBook:
    def __init__(self):
        self.combine_recipes = {
            frozenset(['Iron', 'Fuel']): 'Basic_Engine',
            frozenset(['Copper', 'Stone']): 'Thermal_Core',
            frozenset(['Basic_Engine', 'Thermal_Core']): 'Hybrid_Drive',
            frozenset(['Hybrid_Drive', 'Wood']): 'Aerial_Transport',
            frozenset(['Basic_Engine', 'Wood']): 'Reinforced_Frame',
            frozenset(['Iron', 'Fuel', 'Stone']): 'Steam_Generator',
            frozenset(['Steam_Generator', 'Wood']): 'Steam_Cart',
            frozenset(['Fuel', 'Copper', 'Wood']): 'Copper_Furnace'
        }
        self.decompose_recipes = {v: list(k) for k, v in self.combine_recipes.items()}
    
    def get_combination(self, items: Set[str]) -> Optional[str]:
        return self.combine_recipes.get(frozenset(items), None)
    
    def get_decomposition(self, item: str) -> Optional[List[str]]:
        return self.decompose_recipes.get(item)

class CraftingDomain:
    def __init__(self, allowed_basics: Set[str], recipe_book: RecipeBook):
        self.recipes = recipe_book
        self.allowed_items = allowed_basics | set(recipe_book.combine_recipes.values())
        self.MAX_CAPACITY = 3

    def find_plan(self, target: str) -> List[str]:
        initial_state = (frozenset(), frozenset(self.allowed_items))
        queue = deque([(initial_state, [])])
        visited = set()

        while queue:
            (holding, available), path = queue.popleft()
            
            if target in holding:
                return path
            
            state_key = (holding, available)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            if len(holding) >= self.MAX_CAPACITY:
                for item in holding:
                    new_holding = set(holding)
                    new_holding.remove(item)
                    new_available = set(available)
                    new_available.add(item)
                    new_path = path + [f"drop {item}"]
                    queue.append(((frozenset(new_holding), frozenset(new_available)), new_path))
            
            for item in available:
                if item in self.allowed_items and (len(holding) < self.MAX_CAPACITY or len(holding) >= self.MAX_CAPACITY):
                    new_holding = set(holding)
                    new_holding.add(item)
                    new_available = set(available)
                    new_available.discard(item)
                    if len(new_holding) <= self.MAX_CAPACITY:
                        new_path = path + [f"hold {item}"]
                        queue.append(((frozenset(new_holding), frozenset(new_available)), new_path))
            
            for recipe, result in self.recipes.combine_recipes.items():
                if set(recipe).issubset(holding) and result in self.allowed_items:
                    new_holding = set(holding) - set(recipe)
                    new_holding.add(result)
                    new_available = set(available)
                    new_available.add(result)
                    new_path = path + [f"build {result}"]
                    queue.append(((frozenset(new_holding), frozenset(new_available)), new_path))
        
        return []

class Game:
    def __init__(self):
        self.grid = GridWorld()
        self.agent = Agent()
        self.recipe_book = RecipeBook()
        self.current_plan = []
    
    def automated_crafting_mission(self, target_item: str, max_steps=10000):
        print(f"\n🚀 Starting mission to craft {target_item}!")
        steps = 0
        
        while steps < max_steps and target_item not in self.agent.inventory:
            if steps % 100 == 0 and steps > 0:
                print(f"\n⏱ Step {steps} Report:")
                print(f"📍 Position: ({self.agent.x}, {self.agent.y})")
                print(f"🎒 Inventory ({len(self.agent.inventory)}/{self.agent.max_inventory}): {self.agent.inventory}")
                print("🔍 Discovered Resources:")
                for item in self.grid.initial_resources.keys():
                    count = len(self.agent.discovered_resources[item])
                    print(f"  - {item}: {count} locations")
                if self.current_plan:
                    print(f"📋 Current Plan: {self.current_plan[:3]}... ({len(self.current_plan)} steps remaining)")
                else:
                    print("📋 Current Plan: Exploring for resources")
                print("--------------------------------------------------")
            
            known_basics = {item for item in self.agent.discovered_resources 
                           if item in self.grid.initial_resources}
            
            domain = CraftingDomain(known_basics, self.recipe_book)
            
            if not self.current_plan:
                self.current_plan = domain.find_plan(target_item)
                if not self.current_plan:
                    self.explore_step()
                    steps += 1
                    continue
            
            try:
                self.execute_plan_step()
            except Exception as e:
                self.current_plan = []
            steps += 1  # Critical fix: Always increment steps
        
        if target_item in self.agent.inventory:
            print(f"\n🎉 Successfully crafted {target_item} in {steps} steps!")
            print(f"📍 Final position: ({self.agent.x}, {self.agent.y})")
            print(f"🎒 Final inventory: {self.agent.inventory}")
        else:
            print("\n🔥 Mission failed: Timeout reached")

    def explore_step(self):
        direction = random.choice(['n','s','e','w'])
        dx, dy = {'n':(0,-1),'s':(0,1),'e':(1,0),'w':(-1,0)}[direction]
        if self.agent.move(dx, dy, self.grid):
            self.agent.steps += 1
            self.update_discoveries()

    def update_discoveries(self):
        for item in self.grid.item_locations:
            for x, y in self.grid.item_locations[item]:
                if abs(x-self.agent.x) <=5 and abs(y-self.agent.y) <=5:
                    self.agent.record_discovery(item, x, y)

    def execute_plan_step(self):
        if not self.current_plan:
            return
        
        action = self.current_plan.pop(0)
        
        if action.startswith("hold"):
            self.handle_hold_action(action)
        elif action.startswith("build"):
            self.handle_build_action(action)
        elif action.startswith("drop"):
            self.handle_drop_action(action)

    def handle_hold_action(self, action: str):
        item = action.split()[1]
        if len(self.agent.inventory) >= self.agent.max_inventory:
            raise Exception("Inventory full")
        self.collect_item(item)

    def handle_drop_action(self, action: str):
        item = action.split()[1]
        if item not in self.agent.inventory:
            raise Exception("Item not in inventory")
        success, _ = self.agent.drop_item(self.grid, item)
        if not success:
            raise Exception("Failed to drop item")

    def handle_build_action(self, action: str):
        target = action.split()[1]
        components = self.recipe_book.get_decomposition(target)
        
        if len(self.agent.inventory) + len(components) - 1 > self.agent.max_inventory:
            self.drop_non_essential(components)
        
        self.craft_item(target)

    def collect_item(self, item: str):
        if item not in self.agent.discovered_resources:
            raise Exception("Unknown resource location")
        
        closest = min(self.agent.discovered_resources[item],
                     key=lambda p: abs(p[0]-self.agent.x)+abs(p[1]-self.agent.y))
        self.agent.x, self.agent.y = closest
        success, _ = self.agent.collect_item(self.grid)
        if not success:
            raise Exception("Collection failed")

    def craft_item(self, target: str):
        components = self.recipe_book.get_decomposition(target)
        if not components:
            raise Exception("Unknown recipe")
        
        missing = [c for c in components if c not in self.agent.inventory]
        if missing:
            raise Exception("Missing components")
        
        temp_inv = self.agent.inventory.copy()
        try:
            for c in components:
                temp_inv.remove(c)
        except ValueError:
            raise Exception("Components mismatch")
        
        self.agent.inventory = temp_inv + [target]

    def drop_non_essential(self, needed_components: List[str]):
        for item in self.agent.inventory.copy():
            if item not in needed_components:
                success, _ = self.agent.drop_item(self.grid, item)
                if success:
                    return
        raise Exception("No items to drop")

if __name__ == "__main__":
    game = Game()
    game.automated_crafting_mission("Hybrid_Drive")