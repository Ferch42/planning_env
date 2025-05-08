class GridWorld:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.populate_grid()
    
    def populate_grid(self):
        # Clear grid and place items at specific coordinates
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = None
                
        # Hard-coded item positions
        self.grid[10][10] = 'Iron'    # (10,10)
        self.grid[40][40] = 'Fuel'    # (40,40)
        self.grid[25][25] = 'Copper'  # (25,25)
        self.grid[35][15] = 'Stone'   # (35,15)
        self.grid[15][30] = 'Wood'    # (15,30)

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
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.inventory = []
        self.max_inventory = 5
    
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
                    row.append('@')  # Agent position
                else:
                    item = self.grid.grid[y][x]
                    if item:
                        row.append(item[0].upper())
                    else:
                        row.append('.')
            print(' '.join(row))
    
    def print_status(self):
        self.print_grid()
        print(f"\nPosition: ({self.agent.x}, {self.agent.y})")
        cell_items = self.grid.get_cell_items(self.agent.x, self.agent.y)
        print(f"Current cell item: {cell_items[0] if cell_items else 'Empty'}")
        print(f"Inventory ({len(self.agent.inventory)}/{self.agent.max_inventory}):")
        for item in self.agent.inventory:
            print(f"- {item}")
    
    def combine_items(self, items):
        items_set = set(items)
        combined = self.recipes.get_combination(items_set)
        if not combined:
            return False, "No recipe for these items"
        
        temp_inv = self.agent.inventory.copy()
        try:
            for item in items:
                temp_inv.remove(item)
        except ValueError:
            return False, "Missing required items"
        
        if len(temp_inv) + 1 > self.agent.max_inventory:
            return False, "Not enough space in backpack"
        
        self.agent.inventory = temp_inv
        self.agent.inventory.append(combined)
        return True, f"Created {combined}!"
    
    def decompose_item(self, item):
        components = self.recipes.get_decomposition(item)
        if not components:
            return False, "Can't decompose this item"
        
        if item not in self.agent.inventory:
            return False, "Item not in inventory"
        
        new_size = len(self.agent.inventory) - 1 + len(components)
        if new_size > self.agent.max_inventory:
            return False, "Not enough space to decompose"
        
        self.agent.inventory.remove(item)
        self.agent.inventory.extend(components)
        return True, f"Decomposed into {components}"

    def run(self):
        print("Welcome to Crafting World!")
        print("Available actions: move, collect, combine, break, put, quit")
        print("Map Key: @ = You, . = Empty, Letters = Items")
        print("Item Positions:")
        print("- Iron (10,10)   Fuel (40,40)")
        print("- Copper (25,25) Stone (35,15)")
        print("- Wood (15,30)")
        
        while True:
            self.print_status()
            action = input("\nWhat would you like to do? ").lower().strip()
            
            if action == 'quit':
                print("Thanks for playing!")
                break
            
            elif action == 'move':
                dir_map = {'n': (0, -1), 's': (0, 1), 'e': (1, 0), 'w': (-1, 0)}
                direction = input("Direction (n/s/e/w)? ").lower().strip()
                if direction in dir_map:
                    dx, dy = dir_map[direction]
                    if self.agent.move(dx, dy, self.grid):
                        print(f"Moved {direction}")
                    else:
                        print("Can't move there!")
                else:
                    print("Invalid direction!")
            
            elif action == 'collect':
                success, msg = self.agent.collect_item(self.grid)
                print(msg)
            
            elif action == 'combine':
                items = input("Items to combine (space-separated): ").lower().split()
                success, msg = self.combine_items(items)
                print(msg)
            
            elif action == 'break':
                item = input("Item to break down: ").strip()
                success, msg = self.decompose_item(item)
                print(msg)
            
            elif action == 'put':
                if not self.agent.inventory:
                    print("Inventory is empty!")
                    continue
                item = input("Item to place: ").strip()
                success, msg = self.agent.drop_item(self.grid, item)
                print(msg)
            
            else:
                print("Invalid action!")

if __name__ == "__main__":
    game = Game()
    game.run()