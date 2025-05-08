import random
from collections import defaultdict

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
        self.steps = 0
        self.discovered_resources = defaultdict(set)
        self.visited_cells = set()
    
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
    
    def record_discovery(self, item, x, y):
        self.discovered_resources[item].add((x, y))

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
        print(f"\nItems within capture range (5 cells): {', '.join(capturable) if capturable else 'None'}")
        print(f"Total steps taken: {self.agent.steps}")

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

    def automated_exploration(self, max_steps=10_000):
        directions = ['n', 's', 'e', 'w']
        print("\nStarting automated exploration...")
        total_resources = sum(len(locs) for locs in self.grid.resource_locations.values())
        
        for _ in range(max_steps):
            direction = random.choice(directions)
            dx, dy = {'n': (0, -1), 's': (0, 1), 
                      'e': (1, 0), 'w': (-1, 0)}[direction]
            
            if self.agent.move(dx, dy, self.grid):
                self.agent.steps += 1
                current_pos = (self.agent.x, self.agent.y)
                self.agent.visited_cells.add(current_pos)
                
                for item, locations in self.grid.resource_locations.items():
                    for (x, y) in locations:
                        distance = abs(x - self.agent.x) + abs(y - self.agent.y)
                        if distance <= 5:
                            self.agent.record_discovery(item, x, y)
                
                if self.agent.steps % 50 == 0:
                    self.print_exploration_progress()
                    
                discovered = sum(len(v) for v in self.agent.discovered_resources.values())
                if discovered >= total_resources:
                    print("\nAll resources discovered!")
                    break
        else:
            print("\nReached maximum exploration steps!")
        
        self.print_exploration_summary()

    def print_exploration_progress(self):
        print(f"\nStep {self.agent.steps}:")
        print(f"Visited cells: {len(self.agent.visited_cells)}")
        print("Discovered resources:")
        for resource, locations in self.agent.discovered_resources.items():
            print(f"- {resource}: {len(locations)} locations")

    def print_exploration_summary(self):
        print("\n=== Exploration Summary ===")
        print(f"Total steps taken: {self.agent.steps}")
        print(f"Unique cells visited: {len(self.agent.visited_cells)}")
        print(f"Percentage of grid explored: {len(self.agent.visited_cells)/2500:.1%}")
        print("\nDiscovered resource locations:")
        for resource, locations in self.agent.discovered_resources.items():
            print(f"\n{resource}:")
            for x, y in sorted(locations):
                print(f"({x:>2}, {y:>2})", end=' ')
        print("\n")

    def run(self):
        print("Welcome to Crafting World!")
        print("Choose mode: [1] Manual [2] Automated")
        choice = input("Mode: ").strip()
        
        if choice == '1':
            print("Available actions: move, collect, combine, break, put, capture, goto, quit")
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
                        if self.agent.move(dx, dy, self.grid):
                            print("Moved", direction)
                            self.agent.steps += 1
                        else:
                            print("Can't move there")
                    else:
                        print("Invalid direction")
                elif action == 'collect':
                    _, msg = self.agent.collect_item(self.grid)
                    print(msg)
                    self.agent.steps += 1
                elif action == 'combine':
                    items = input("Items to combine: ").split()
                    _, msg = self.combine_items(items)
                    print(msg)
                    self.agent.steps += 1
                elif action == 'break':
                    _, msg = self.decompose_item(input("Item to decompose: ").strip())
                    print(msg)
                    self.agent.steps += 1
                elif action == 'put':
                    _, msg = self.agent.drop_item(self.grid, input("Item to place: ").strip())
                    print(msg)
                    self.agent.steps += 1
                elif action == 'goto':
                    try:
                        x = int(input("Enter X coordinate (0-49): "))
                        y = int(input("Enter Y coordinate (0-49): "))
                        if 0 <= x < 50 and 0 <= y < 50:
                            distance = abs(x - self.agent.x) + abs(y - self.agent.y)
                            self.agent.steps += distance
                            self.agent.x = x
                            self.agent.y = y
                            print(f"Teleported to ({x}, {y})")
                        else:
                            print("Coordinates must be between 0-49")
                    except ValueError:
                        print("Invalid coordinates - must be numbers")
                elif action == 'capture':
                    self.agent.steps += 1
                    if len(self.agent.inventory) >= self.agent.max_inventory:
                        print("Inventory full")
                    else:
                        item = input("Item to capture: ").strip()
                        target = self.find_nearby_item(item)
                        if target:
                            target_x, target_y, _ = target
                            distance = abs(target_x - self.agent.x) + abs(target_y - self.agent.y)
                            self.agent.steps += distance
                            self.agent.x = target_x
                            self.agent.y = target_y
                            success, msg = self.agent.collect_item(self.grid)
                            print(msg)
                            if success:
                                self.agent.steps += 1
                        else:
                            print(f"No {item} within 5 cells")
                else:
                    print("Invalid command")
        elif choice == '2':
            self.automated_exploration()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    Game().run()