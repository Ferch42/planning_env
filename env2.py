import random
import os

# --- Game Configuration ---
GRID_SIZE = 10 # Try values like 10, 15, or 5

# Terrain types and their visual representation
TERRAIN_TYPES = {
    'EMPTY': {'id': 'EMPTY', 'emoji': ' ', 'resource': None, 'message': "Just an empty space."},
    'FOREST': {'id': 'FOREST', 'emoji': 'F', 'resource': 'wood', 'message': "You are in a dense forest."},
    'MINE': {'id': 'MINE', 'emoji': 'M', 'resource': 'stone', 'message': "You've found a promising mine."},
    'LAKE': {'id': 'LAKE', 'emoji': 'L', 'resource': 'water', 'message': "You are by a calm lake."}
}
AGENT_EMOJI = '🧍'

# --- Game State ---
grid = []  # 2D list representing the world
agent_position = {'x': 0, 'y': 0}  # Agent's current position
inventory = {
    'wood': 0,
    'stone': 0,
    'water': 0
}
last_message = "Welcome! Explore the newly arranged world."

# --- Helper Functions ---
def clear_console():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Game Logic ---

def initialize_grid():
    """
    Initializes the game grid with specified terrain concentrations.
    - Forests ('F') in the first two rows.
    - Mines ('M') in the last two rows.
    - Lakes ('L') in the top-right border area.
    - Agent starts at (0,0) which is EMPTY.
    """
    global grid, agent_position
    grid = []

    # Step 1: Initialize all cells to a base type, e.g., EMPTY. Using .copy() for dictionaries.
    for _ in range(GRID_SIZE):
        grid.append([TERRAIN_TYPES['EMPTY'].copy() for _ in range(GRID_SIZE)])

    # Step 2: Place Forests in the first two rows (y=0, y=1)
    # min(2, GRID_SIZE) ensures we don't go out of bounds for small grids.
    for y_coord in range(min(2, GRID_SIZE)):
        for x_coord in range(GRID_SIZE):
            if random.random() < 0.80:  # 80% chance of Forest in these rows
                grid[y_coord][x_coord] = TERRAIN_TYPES['FOREST'].copy()

    # Step 3: Place Mines in the last two rows
    # max(0, GRID_SIZE - 2) handles the starting row for mines correctly.
    for y_coord in range(max(0, GRID_SIZE - 2), GRID_SIZE):
        for x_coord in range(GRID_SIZE):
            if random.random() < 0.80:  # 80% chance of Mine in these rows
                grid[y_coord][x_coord] = TERRAIN_TYPES['MINE'].copy()
    
    # Step 4: Place Lake in the top right border area
    if GRID_SIZE > 0:
        # Top-right corner cell
        grid[0][GRID_SIZE - 1] = TERRAIN_TYPES['LAKE'].copy()
        
        # Extend along top border (to the left from corner)
        if GRID_SIZE - 2 >= 0: # Cell (0, GRID_SIZE-2)
            if random.random() < 0.9: grid[0][GRID_SIZE - 2] = TERRAIN_TYPES['LAKE'].copy()
        if GRID_SIZE - 3 >= 0: # Cell (0, GRID_SIZE-3)
            if random.random() < 0.7: grid[0][GRID_SIZE - 3] = TERRAIN_TYPES['LAKE'].copy()
        
        # Extend along right border (downwards from corner)
        if GRID_SIZE > 1: # Cell (1, GRID_SIZE-1)
             if random.random() < 0.9: grid[1][GRID_SIZE - 1] = TERRAIN_TYPES['LAKE'].copy()
        if GRID_SIZE > 2: # Cell (2, GRID_SIZE-1) (check y_coord < GRID_SIZE)
             if 2 < GRID_SIZE and random.random() < 0.7: grid[2][GRID_SIZE - 1] = TERRAIN_TYPES['LAKE'].copy()

    # Step 5: Fill remaining unspecified areas (middle rows)
    # These are rows from y=2 to y=GRID_SIZE-3 (exclusive of last two mine rows)
    # This loop runs if GRID_SIZE >= 5.
    for y_coord in range(2, GRID_SIZE - 2):
        for x_coord in range(GRID_SIZE):
            # Only fill if the cell hasn't been set by a more specific rule (e.g. part of the lake border)
            if grid[y_coord][x_coord]['id'] == 'EMPTY':
                rand_val = random.random()
                if rand_val < 0.05: # Small chance of Forest
                    grid[y_coord][x_coord] = TERRAIN_TYPES['FOREST'].copy()
                elif rand_val < 0.10: # Small chance of Mine
                    grid[y_coord][x_coord] = TERRAIN_TYPES['MINE'].copy()
                elif rand_val < 0.18: # Slightly higher chance of Lake in middle
                    grid[y_coord][x_coord] = TERRAIN_TYPES['LAKE'].copy()
                # else it remains EMPTY (82% chance)

    # Step 6: Ensure agent's starting position (0,0) is EMPTY.
    # This overrides any Forest/Lake that might have been placed at (0,0).
    if GRID_SIZE > 0:
        grid[0][0] = TERRAIN_TYPES['EMPTY'].copy()

    agent_position = {'x': 0, 'y': 0}


def display_grid():
    """Renders the entire grid and the agent to the console."""
    clear_console()
    print("--- Grid World Adventure ---")
    print("+" + "---+" * GRID_SIZE) # Top border of the grid

    for y_coord in range(GRID_SIZE):
        row_str = "|" # Start of a row line
        for x_coord in range(GRID_SIZE):
            if x_coord == agent_position['x'] and y_coord == agent_position['y']:
                # Agent's cell, ensure it's centered like other characters
                row_str += f" {AGENT_EMOJI} |"
            else:
                # Terrain cell
                char_to_display = grid[y_coord][x_coord]['emoji']
                row_str += f" {char_to_display} |" 
        print(row_str)
        print("+" + "---+" * GRID_SIZE) # Row separator / Bottom border of grid cells
    print(f"Agent ( {AGENT_EMOJI} ) at: ({agent_position['x']}, {agent_position['y']})")


def display_inventory():
    """Displays the current inventory."""
    print("\n--- Inventory ---")
    print(f"Wood 🪵: {inventory['wood']}")
    print(f"Stone 🪨: {inventory['stone']}")
    print(f"Water 💧: {inventory['water']}")

def display_messages():
    """Displays game messages."""
    global last_message
    current_tile_info = grid[agent_position['y']][agent_position['x']]['message']
    print(f"\n--- Messages ---")
    if last_message: 
        print(last_message)
    print(f"Current: {current_tile_info}")
    last_message = "" 

def move_agent(dx, dy):
    """Handles agent movement."""
    global agent_position, last_message
    new_x = agent_position['x'] + dx
    new_y = agent_position['y'] + dy

    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
        agent_position['x'] = new_x
        agent_position['y'] = new_y
        # Message about new tile will be shown by display_messages()
    else:
        last_message = "Cannot move outside the world! You hit an invisible wall."

def collect_resource():
    """Handles resource collection."""
    global last_message
    current_tile = grid[agent_position['y']][agent_position['x']]
    if current_tile and current_tile['resource']:
        resource_name = current_tile['resource']
        inventory[resource_name] += 1
        last_message = f"Collected 1 {resource_name}! You now have {inventory[resource_name]}."
    else:
        last_message = "Nothing to collect here."

def get_action_on_tile():
    """Returns a string describing possible actions on the current tile."""
    current_tile = grid[agent_position['y']][agent_position['x']]
    if current_tile and current_tile['resource']:
        return f"Press 'c' to collect {current_tile['resource']}."
    return ""

# --- Main Game Loop ---
def game_loop():
    """Main loop for the game."""
    global last_message
    initialize_grid()

    while True:
        display_grid()
        display_inventory()
        display_messages()

        print("\n--- Actions ---")
        print("Move: (w)up, (s)down, (a)left, (d)right")
        collect_action_text = get_action_on_tile()
        if collect_action_text:
            print(collect_action_text)
        print("Press 'q' to quit.")

        action = input("> ").lower().strip()

        if action == 'w':
            move_agent(0, -1)
        elif action == 's':
            move_agent(0, 1)
        elif action == 'a':
            move_agent(-1, 0)
        elif action == 'd':
            move_agent(1, 0)
        elif action == 'c':
            collect_resource()
        elif action == 'q':
            last_message = "Thanks for playing Grid World Adventure!"
            display_grid() 
            display_inventory()
            display_messages()
            print("\nQuitting game...")
            break
        else:
            last_message = "Invalid action. Try again."

if __name__ == "__main__":
    try:
        game_loop()
    except KeyboardInterrupt: 
        clear_console()
        print("\nGame interrupted by user. Exiting...")
