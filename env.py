class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.populate_grid()
    
    def populate_grid(self):
        # Precisely placed components for all recipes
        self.grid[0] = ['stick', 'stone', 'wood', 'leaf', 'flower']
        self.grid[1] = ['flower', 'coal', 'iron', 'rubber', 'rubber']
    
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

# ... (RecipeBook class remains unchanged from previous version)

class Game:
    def __init__(self):
        self.grid = GridWorld()
        self.agent = Agent()
        self.recipes = RecipeBook()
    
    def print_grid(self):
        print("\nGrid World:")
        for y in range(self.grid.height):
            row = []
            for x in range(self.grid.width):
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
    
    # ... (combine_items and decompose_item methods remain unchanged)

    def run(self):
        print("Welcome to GridWorld!")
        print("Available actions: move, collect, combine, break, put, quit")
        print("Map Key: @ = You, . = Empty, Letters = Items")
        print("Fixed Grid Contains:")
        print("Row 0: Stick, Stone, Wood, Leaf, Flower")
        print("Row 1: Flower, Coal, Iron, Rubber, Rubber")
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
                item = input("Item to break down: ").lower().strip()
                success, msg = self.decompose_item(item)
                print(msg)
            
            elif action == 'put':
                if not self.agent.inventory:
                    print("Inventory is empty!")
                    continue
                item = input("Item to place: ").lower().strip()
                success, msg = self.agent.drop_item(self.grid, item)
                print(msg)
            
            else:
                print("Invalid action!")

if __name__ == "__main__":
    game = Game()
    game.run()