import random

class GridWorld:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.resource_locations = {
            'Iron': [(5, 5), (45, 45)],
            'Fuel': [(5, 45), (45, 5)],
            'Copper': [(15, 10), (35, 40)],
            'Stone': [(10, 25), (40, 25)],
            'Wood': [(20, 20), (30, 30)]
        }
        self.populate_grid()
    
    def populate_grid(self):
        for item, positions in self.resource_locations.items():
            for x, y in positions:
                self.grid[y][x] = item

    def get_cell_items(self, x, y):
        return [self.grid[y][x]] if self.grid[y][x] is not None else []
    
    def remove_item(self, x, y):
        if self.grid[y][x] is not None:
            removed_item = self.grid[y][x]
            self.grid[y][x] = None
            return removed_item
        return None
    
    def place_item(self, x, y, item):
        if self.grid[y][x] is None:
            self.grid[y][x] = item
            return True
        return False

class Agent:
    def __init__(self, x=25, y=25):
        self.x = x
        self.y = y
        self.inventory = []
        self.max_inventory = 3
    
    def move(self, dx, dy, grid):
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
            self.x, self.y = new_x, new_y
            return True
        return False
    
    def collect_item(self, grid):
        if len(self.inventory) >= self.max_inventory:
            return False, "Backpack full!"
        items = grid.get_cell_items(self.x, self.y)
        if not items:
            return False, "No items here!"
        item = grid.remove_item(self.x, self.y)
        if item:
            self.inventory.append(item)
            return True, f"Collected {item}"
        return False, "Collection failed"
    
    def drop_item(self, grid, item):
        if item not in self.inventory:
            return False, "Item not in inventory"
        if not grid.place_item(self.x, self.y, item):
            return False, "Cell is not empty"
        self.inventory.remove(item)
        return True, f"Placed {item} at ({self.x}, {self.y})"

class RecipeBook:
    def __init__(self):
        self.combine_recipes = {
            frozenset(['Iron', 'Fuel']): 'Basic_Engine',
            frozenset(['Copper', 'Stone']): 'Thermal_Core',
            frozenset(['Basic_Engine', 'Thermal_Core']): 'Hybrid_Drive',
            frozenset(['Hybrid_Drive', 'Wood']): 'Aerial_Transport',
            frozenset(['Basic_Engine', 'Wood']): 'Reinforced_Frame'
        }
        self.decompose_recipes = {
            'Basic_Engine': ['Iron', 'Fuel'],
            'Thermal_Core': ['Copper', 'Stone'],
            'Hybrid_Drive': ['Basic_Engine', 'Thermal_Core'],
            'Aerial_Transport': ['Hybrid_Drive', 'Wood'],
            'Reinforced_Frame': ['Basic_Engine', 'Wood']
        }
    
    def get_combination(self, items):
        return self.combine_recipes.get(frozenset(items), None)
    
    def get_decomposition(self, item):
        return self.decompose_recipes.get(item, None)

class Game:
    def __init__(self):
        self.grid = GridWorld()
        self.agent = Agent()
        self.recipes = RecipeBook()
    
    def find_nearby_item(self, item_name):
        CAPTURE_RANGE = 5
        targets = []
        for x, y in self.grid.resource_locations.get(item_name, []):
            distance = abs(x - self.agent.x) + abs(y - self.agent.y)
            if distance <= CAPTURE_RANGE:
                targets.append((x, y, distance))
        return min(targets, key=lambda t: t[2]) if targets else None

    def capture_item(self, item_name):
        target = self.find_nearby_item(item_name)
        if not target:
            return False, f"No {item_name} within 5 cells"
        target_x, target_y, _ = target
        self.agent.x, self.agent.y = target_x, target_y
        success, msg = self.agent.collect_item(self.grid)
        return (success, msg) if success else (False, f"Capture failed: {msg}")

    def get_capturable_items(self):
        capturable = set()
        for item in self.grid.resource_locations:
            if self.find_nearby_item(item):
                capturable.add(item)
        return sorted(capturable)

    def print_grid(self):
        print("\nLocal View (5x5 around agent):")
        min_y = max(0, self.agent.y-2)
        max_y = min(self.grid.height, self.agent.y+3)
        min_x = max(0, self.agent.x-2)
        max_x = min(self.grid.width, self.agent.x+3)
        for y in range(min_y, max_y):
            row = []
            for x in range(min_x, max_x):
                if x == self.agent.x and y == self.agent.y:
                    row.append('@')
                else:
                    item = self.grid.grid[y][x]
                    row.append(item[0].upper() if item else '.')
            print(' '.join(row))
    
    def print_status(self):
        self.print_grid()
        print(f"\nPosition: ({self.agent.x}, {self.agent.y})")
        print(f"Inventory ({len(self.agent.inventory)}/{self.agent.max_inventory}):")
        for item in self.agent.inventory:
            print(f"- {item}")
        capturable = self.get_capturable_items()
        print("\nItems within capture range (5 cells):", 
              ', '.join(capturable) if capturable else "None")

    def combine_items(self, items):
        combined = self.recipes.get_combination(set(items))
        if not combined:
            return False, "No recipe for these items"
        temp_inv = self.agent.inventory.copy()
        try:
            for item in items:
                temp_inv.remove(item)
        except ValueError:
            return False, "Missing items"
        if len(temp_inv) + 1 > self.agent.max_inventory:
            return False, "Inventory full"
        self.agent.inventory = temp_inv + [combined]
        return True, f"Created {combined}!"

    def decompose_item(self, item):
        components = self.recipes.get_decomposition(item)
        if not components:
            return False, "Can't decompose"
        if item not in self.agent.inventory:
            return False, "Not in inventory"
        if len(self.agent.inventory) + len(components) - 1 > self.agent.max_inventory:
            return False, "Not enough space"
        self.agent.inventory.remove(item)
        self.agent.inventory.extend(components)
        return True, f"Decomposed into {', '.join(components)}"

    def run(self):
        print("Welcome to Crafting World!")
        print("Available actions: move, collect, combine, break, put, capture, quit")
        while True:
            self.print_status()
            action = input("\nAction: ").lower().strip()
            
            if action == 'quit':
                print("Thanks for playing!")
                break
            elif action == 'move':
                dir_map = {'n': (0, -1), 's': (0, 1), 'e': (1, 0), 'w': (-1, 0)}
                direction = input("Direction (n/s/e/w)? ").lower().strip()
                if direction in dir_map:
                    dx, dy = dir_map[direction]
                    print("Moved", direction if self.agent.move(dx, dy, self.grid) else "Can't move there")
                else: print("Invalid direction")
            elif action == 'collect':
                print(self.agent.collect_item(self.grid)[1])
            elif action == 'combine':
                items = input("Items to combine: ").split()
                print(self.combine_items(items)[1])
            elif action == 'break':
                print(self.decompose_item(input("Item to decompose: ").strip())[1])
            elif action == 'put':
                print(self.agent.drop_item(self.grid, input("Item to place: ").strip())[1])
            elif action == 'capture':
                if len(self.agent.inventory) >= self.agent.max_inventory:
                    print("Inventory full")
                else:
                    print(self.capture_item(input("Item to capture: ").strip())[1])
            else:
                print("Invalid command")

if __name__ == "__main__":
    Game().run()