import numpy as np
import random
from enum import Enum

class ObjectType(Enum):
    EMPTY = 0
    KEY = 1
    TREASURE = 2
    FOOD = 3
    TOOL = 4
    WEAPON = 5
    POTION = 6
    BOOK = 7
    CRYSTAL = 8
    GEM = 9
    COIN = 10

class GridWorld:
    def __init__(self, num_rooms=16, room_size=5):
        self.num_rooms = num_rooms
        self.room_size = room_size
        self.rooms_per_side = int(np.sqrt(num_rooms))
        
        # Account for walls between rooms
        self.grid_size = self.rooms_per_side * room_size + (self.rooms_per_side - 1)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Store table positions
        self.table_positions = {}
        
        # Initialize agent at position (1,1)
        self.agent_pos = (1, 1)
        self.agent_inventory = None
        
        self._build_rooms()
        self._assign_objects()
    
    def _build_rooms(self):
        """Build interior walls between rooms"""
        # Add interior walls
        for i in range(1, self.rooms_per_side):
            wall_pos = i * (self.room_size + 1) - 1
            self.grid[wall_pos, :] = 1  # Horizontal walls
            self.grid[:, wall_pos] = 1  # Vertical walls
        
        # Add doors between rooms
        door_pos = self.room_size // 2
        for i in range(1, self.rooms_per_side):
            for j in range(self.rooms_per_side):
                wall_row = i * (self.room_size + 1) - 1
                wall_col = i * (self.room_size + 1) - 1
                
                # Horizontal doors
                door_col = j * (self.room_size + 1) + door_pos
                self.grid[wall_row, door_col] = 0
                
                # Vertical doors
                door_row = j * (self.room_size + 1) + door_pos
                self.grid[door_row, wall_col] = 0
    
    def _assign_objects(self):
        """Assign multiple object types to room centers"""
        # Use all object types except EMPTY
        object_types = [obj.value for obj in ObjectType if obj != ObjectType.EMPTY]
        
        for room_x in range(self.rooms_per_side):
            for room_y in range(self.rooms_per_side):
                center_x = room_x * (self.room_size + 1) + self.room_size // 2
                center_y = room_y * (self.room_size + 1) + self.room_size // 2
                
                self.table_positions[(center_x, center_y)] = (room_x, room_y)
                
                # 90% chance to have an object with random type
                if random.random() > 0.1:
                    obj_type = random.choice(object_types)
                    self.grid[center_x, center_y] = obj_type
    
    def step(self, action):
        """Execute an action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=TOGGLE)"""
        x, y = self.agent_pos
        
        if action == 0 and x > 0 and self.grid[x-1, y] != 1:  # UP
            self.agent_pos = (x-1, y)
        elif action == 1 and x < self.grid_size-1 and self.grid[x+1, y] != 1:  # DOWN
            self.agent_pos = (x+1, y)
        elif action == 2 and y > 0 and self.grid[x, y-1] != 1:  # LEFT
            self.agent_pos = (x, y-1)
        elif action == 3 and y < self.grid_size-1 and self.grid[x, y+1] != 1:  # RIGHT
            self.agent_pos = (x, y+1)
        elif action == 4:  # TOGGLE_OBJECT
            self._toggle_object()
    
    def _toggle_object(self):
        """Pick up or put down object if agent is at a table"""
        if self.agent_pos in self.table_positions:
            if self.agent_inventory is None:
                # Pick up object if there is one
                if self.grid[self.agent_pos] >= 2:
                    self.agent_inventory = self.grid[self.agent_pos]
                    self.grid[self.agent_pos] = 0
            else:
                # Put down object if table is empty
                if self.grid[self.agent_pos] == 0:
                    self.grid[self.agent_pos] = self.agent_inventory
                    self.agent_inventory = None
    
    def get_state(self):
        """Return current grid state with agent position marked as -1"""
        state = self.grid.copy()
        x, y = self.agent_pos
        state[x, y] = -1  # Mark agent position
        return state
    
    def render(self):
        """Display the current state"""
        print(self.get_state())


# Create environment with 25 rooms and multiple object types
world = GridWorld(num_rooms=16, room_size=5)

# Show initial state
print("Initial state (25 rooms, multiple object types):")
world.render()

# Example actions
actions = [3, 1, 4]  # RIGHT, DOWN, TOGGLE

for i, action in enumerate(actions):
    print(f"\nStep {i+1}, Action: {action}")
    world.step(action)
    world.render()
    if world.agent_inventory:
        obj_name = ObjectType(world.agent_inventory).name
        print(f"Agent carrying: {obj_name} ({world.agent_inventory})")