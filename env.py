class GridWorld:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.populate_grid()
    
    def populate_grid(self):
        import random
        items = ['stick', 'stone', 'wood', 'leaf', 'flower', 'coal', 'iron', 'rubber']
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < 0.4:  # 40% chance of an item
                    self.grid[y][x] = random.choice(items)
    
    def get_cell_items(self, x, y):
        return [self.grid[y][x]] if self.grid[y][x] is not None else []
    
    def remove_item(self, x, y):
        if self.grid[y][x] is not None:
            removed_item = self.grid[y][x]
            self.grid[y][x] = None
            return removed_item
        return None

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

class RecipeBook:
    def __init__(self):
        self.combine_recipes = {
            frozenset(['stick', 'stone']): 'axe',
            frozenset(['wood', 'axe']): 'firewood',
            frozenset(['coal', 'iron']): 'steel',
            frozenset(['steel', 'rubber']): 'machine_part',
            frozenset(['machine_part', 'steel', 'rubber']): 'robot',
            # New recipes
            frozenset(['flower', 'leaf']): 'herbal_remedy',
            frozenset(['herbal_remedy', 'flower']): 'healing_potion',
            frozenset(['firewood', 'stone']): 'campfire',
            frozenset(['stick', 'coal']): 'torch',
            frozenset(['steel', 'stick']): 'sword',
            frozenset(['wood', 'steel']): 'shield',
            frozenset(['stick', 'leaf']): 'rope',
            frozenset(['wood', 'rope']): 'bridge',
            frozenset(['stick', 'rope']): 'ladder',
            frozenset(['rubber', 'steel']): 'electrical_wire',
            frozenset(['machine_part', 'steel', 'electrical_wire']): 'engine',
            frozenset(['engine', 'robot', 'steel']): 'flying_machine'
        }
        self.decompose_recipes = {
            'axe': ['stick', 'stone'],
            'firewood': ['wood', 'axe'],
            'steel': ['coal', 'iron'],
            'machine_part': ['steel', 'rubber'],
            'robot': ['machine_part', 'steel', 'rubber'],
            # New decompositions
            'herbal_remedy': ['flower', 'leaf'],
            'healing_potion': ['herbal_remedy', 'flower'],
            'campfire': ['firewood', 'stone'],
            'torch': ['stick', 'coal'],
            'sword': ['steel', 'stick'],
            'shield': ['wood', 'steel'],
            'rope': ['stick', 'leaf'],
            'bridge': ['wood', 'rope'],
            'ladder': ['stick', 'rope'],
            'electrical_wire': ['rubber', 'steel'],
            'engine': ['machine_part', 'steel', 'electrical_wire'],
            'flying_machine': ['engine', 'robot', 'steel']
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
    
    def print_status(self):
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
        print("Welcome to GridWorld!")
        print("Available actions: move, collect, combine, break, quit")
        print("Some example recipes:")
        print("- stick + stone = axe")
        print("- flower + leaf = herbal_remedy")
        print("- steel + stick = sword")
        print("- robot + engine + steel = flying_machine")
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
            
            else:
                print("Invalid action!")

if __name__ == "__main__":
    game = Game()
    game.run()