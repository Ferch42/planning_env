import gym
from gym import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

class ObjectType(Enum):
    EMPTY = 0
    KEY = 1
    TREASURE = 2
    AGENT = 3
    OBSTACLE = 4
    FOOD = 5
    WEAPON = 6

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}
    
    def __init__(self, room_size=5, room_rows=4, room_cols=4):
        super(GridWorldEnv, self).__init__()
        
        self.room_size = room_size
        self.room_rows = room_rows
        self.room_cols = room_cols
        self.width = room_cols * room_size
        self.height = room_rows * room_size
        
        self.num_rooms_x = room_cols
        self.num_rooms_y = room_rows
        
        # Define action space: 0=up, 1=right, 2=down, 3=left, 4=pickup, 5=drop
        self.action_space = spaces.Discrete(6)
        
        # New observation space with predicate information
        # We'll use a Dict space with string representations of predicates
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=max(self.width, self.height), 
                                  shape=(2,), dtype=np.int32),
            'inventory': spaces.Box(low=0, high=len(ObjectType)-1, 
                                  shape=(1,), dtype=np.int32),
            'predicates': spaces.Text(max_length=1000)  # String representation of predicates
        })
        
        # Initialize environment state
        self.grid = None
        self.agent_pos = None
        self.rooms = {}
        self.objects = []  # List of (room_id, x, y, object_type)
        self.doors = defaultdict(set)  # Maps (x,y) to set of allowed directions
        self.door_positions = set()  # Store door positions for rendering
        self.door_cells = set()  # Cells adjacent to doors
        self.inventory = ObjectType.EMPTY  # What the agent is carrying
        
        self.reset()
        
    def _get_room_id(self, x, y):
        """Get room ID for a given position"""
        room_x = x // self.room_size
        room_y = y // self.room_size
        return f"room_{room_x}_{room_y}"
    
    def _get_room_bounds(self, room_id):
        """Get the boundaries of a room"""
        parts = room_id.split('_')
        room_x = int(parts[1])
        room_y = int(parts[2])
        x_start = room_x * self.room_size
        y_start = room_y * self.room_size
        x_end = min((room_x + 1) * self.room_size, self.width)
        y_end = min((room_y + 1) * self.room_size, self.height)
        return x_start, y_start, x_end, y_end
    
    def _get_random_empty_position_in_room(self, room_id, avoid_door_areas=True):
        """Get a random empty position within a room, avoiding door areas"""
        x_start, y_start, x_end, y_end = self._get_room_bounds(room_id)
        
        # Try to find an empty position
        for _ in range(200):  # Increased attempts for larger rooms
            x = np.random.randint(x_start, x_end)
            y = np.random.randint(y_start, y_end)
            
            # Skip if this is a door cell and we're avoiding door areas
            if avoid_door_areas and (x, y) in self.door_cells:
                continue
                
            if self.grid[y, x] == ObjectType.EMPTY.value:
                return x, y
        
        # If no empty spot found, try without door avoidance
        for _ in range(100):
            x = np.random.randint(x_start, x_end)
            y = np.random.randint(y_start, y_end)
            if self.grid[y, x] == ObjectType.EMPTY.value:
                return x, y
        
        # Last resort: center of room
        return (x_start + x_end) // 2, (y_start + y_end) // 2
    
    def _create_minimal_connections(self):
        """Create minimal connections to ensure all rooms are reachable"""
        # Create a grid of rooms
        room_grid = [[f"room_{x}_{y}" for y in range(self.num_rooms_y)] for x in range(self.num_rooms_x)]
        
        # Start with room_0_0 and build connections
        visited = set()
        to_visit = [(0, 0)]
        connections = []
        
        while to_visit:
            current_x, current_y = to_visit.pop(0)
            if (current_x, current_y) in visited:
                continue
                
            visited.add((current_x, current_y))
            
            # Get unvisited neighbors
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.num_rooms_x and 0 <= ny < self.num_rooms_y:
                    neighbors.append((nx, ny, dx, dy))
            
            # Randomly shuffle neighbors
            np.random.shuffle(neighbors)
            
            # Connect to 1-2 neighbors (not all)
            num_connections = np.random.randint(1, 3)
            connected = 0
            
            for nx, ny, dx, dy in neighbors:
                if (nx, ny) not in visited and connected < num_connections:
                    connections.append((current_x, current_y, nx, ny, dx, dy))
                    to_visit.append((nx, ny))
                    connected += 1
            
            # If we didn't connect to any new rooms, force at least one connection
            if connected == 0 and neighbors:
                nx, ny, dx, dy = neighbors[0]
                connections.append((current_x, current_y, nx, ny, dx, dy))
                to_visit.append((nx, ny))
        
        return connections
    
    def _place_doors_between_rooms(self):
        """Place doors to ensure all rooms are reachable with limited connections"""
        self.doors.clear()
        self.door_positions.clear()
        self.door_cells.clear()
        
        # First, create minimal connections to ensure all rooms are reachable
        connections = self._create_minimal_connections()
        
        # Place doors for the connections
        for current_x, current_y, next_x, next_y, dx, dy in connections:
            if dx == 1:  # Right connection
                # Choose a random vertical position for the door
                y = np.random.randint(current_y * self.room_size, (current_y + 1) * self.room_size)
                x = (current_x + 1) * self.room_size
                
                # Add door in both directions
                self.doors[(x-1, y)].add((1, 0))  # Right direction
                self.doors[(x, y)].add((-1, 0))   # Left direction
                self.door_positions.add((x-1, y, 1, 0))
                self.door_positions.add((x, y, -1, 0))
                
                # Mark door cells (cells adjacent to doors)
                self.door_cells.add((x-1, y))
                self.door_cells.add((x, y))
                
            elif dx == -1:  # Left connection
                # Choose a random vertical position for the door
                y = np.random.randint(current_y * self.room_size, (current_y + 1) * self.room_size)
                x = current_x * self.room_size
                
                # Add door in both directions
                self.doors[(x-1, y)].add((1, 0))  # Right direction
                self.doors[(x, y)].add((-1, 0))   # Left direction
                self.door_positions.add((x-1, y, 1, 0))
                self.door_positions.add((x, y, -1, 0))
                
                # Mark door cells
                self.door_cells.add((x-1, y))
                self.door_cells.add((x, y))
                
            elif dy == 1:  # Down connection
                # Choose a random horizontal position for the door
                x = np.random.randint(current_x * self.room_size, (current_x + 1) * self.room_size)
                y = (current_y + 1) * self.room_size
                
                # Add door in both directions
                self.doors[(x, y-1)].add((0, 1))  # Down direction
                self.doors[(x, y)].add((0, -1))   # Up direction
                self.door_positions.add((x, y-1, 0, 1))
                self.door_positions.add((x, y, 0, -1))
                
                # Mark door cells
                self.door_cells.add((x, y-1))
                self.door_cells.add((x, y))
                
            elif dy == -1:  # Up connection
                # Choose a random horizontal position for the door
                x = np.random.randint(current_x * self.room_size, (current_x + 1) * self.room_size)
                y = current_y * self.room_size
                
                # Add door in both directions
                self.doors[(x, y-1)].add((0, 1))  # Down direction
                self.doors[(x, y)].add((0, -1))   # Up direction
                self.door_positions.add((x, y-1, 0, 1))
                self.door_positions.add((x, y, 0, -1))
                
                # Mark door cells
                self.door_cells.add((x, y-1))
                self.door_cells.add((x, y))
        
        # Add a few extra random doors for variety, but not too many
        extra_doors = 0
        max_extra_doors = (self.num_rooms_x * self.num_rooms_y) // 4
        
        # Place vertical doors (between rooms horizontally)
        for room_x in range(self.num_rooms_x - 1):
            for room_y in range(self.num_rooms_y):
                # Skip if this connection is already made
                is_connected = any(
                    (room_x == cx and room_y == cy and room_x+1 == nx and room_y == ny) or
                    (room_x+1 == cx and room_y == cy and room_x == nx and room_y == ny)
                    for cx, cy, nx, ny, _, _ in connections
                )
                
                if not is_connected and extra_doors < max_extra_doors and np.random.random() < 0.15:
                    # Choose 1 random vertical position for door in this boundary
                    y = np.random.randint(room_y * self.room_size, (room_y + 1) * self.room_size)
                    x = (room_x + 1) * self.room_size
                    
                    if y < self.height:
                        # Add door in both directions
                        self.doors[(x-1, y)].add((1, 0))  # Right direction
                        self.doors[(x, y)].add((-1, 0))   # Left direction
                        self.door_positions.add((x-1, y, 1, 0))
                        self.door_positions.add((x, y, -1, 0))
                        
                        # Mark door cells
                        self.door_cells.add((x-1, y))
                        self.door_cells.add((x, y))
                        extra_doors += 1
        
        # Place horizontal doors (between rooms vertically)
        for room_x in range(self.num_rooms_x):
            for room_y in range(self.num_rooms_y - 1):
                # Skip if this connection is already made
                is_connected = any(
                    (room_x == cx and room_y == cy and room_x == nx and room_y+1 == ny) or
                    (room_x == cx and room_y+1 == cy and room_x == nx and room_y == ny)
                    for cx, cy, nx, ny, _, _ in connections
                )
                
                if not is_connected and extra_doors < max_extra_doors and np.random.random() < 0.15:
                    # Choose 1 random horizontal position for door in this boundary
                    x = np.random.randint(room_x * self.room_size, (room_x + 1) * self.room_size)
                    y = (room_y + 1) * self.room_size
                    
                    if x < self.width:
                        # Add door in both directions
                        self.doors[(x, y-1)].add((0, 1))  # Down direction
                        self.doors[(x, y)].add((0, -1))   # Up direction
                        self.door_positions.add((x, y-1, 0, 1))
                        self.door_positions.add((x, y, 0, -1))
                        
                        # Mark door cells
                        self.door_cells.add((x, y-1))
                        self.door_cells.add((x, y))
                        extra_doors += 1
    
    def _is_door_between(self, from_pos, to_pos):
        """Check if there's a door allowing movement from from_pos to to_pos"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Check if there's a door at from_pos allowing movement in (dx, dy) direction
        return (dx, dy) in self.doors.get((from_pos[0], from_pos[1]), set())
    
    def _initialize_rooms(self):
        """Initialize rooms with unique IDs and basic structure"""
        self.rooms = {}
        for x in range(self.num_rooms_x):
            for y in range(self.num_rooms_y):
                room_id = f"room_{x}_{y}"
                self.rooms[room_id] = {
                    'id': room_id,
                    'bounds': self._get_room_bounds(room_id),
                    'objects': []
                }
    
    def _place_objects(self):
        """Place objects randomly in different rooms - avoiding door areas"""
        self.objects = []
        
        # Place keys in random rooms (3 keys)
        for _ in range(3):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            key_x, key_y = self._get_random_empty_position_in_room(room_id, avoid_door_areas=True)
            self._add_object(room_id, key_x, key_y, ObjectType.KEY)
        
        # Place treasures in random rooms (2 treasures)
        for _ in range(2):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            treasure_x, treasure_y = self._get_random_empty_position_in_room(room_id, avoid_door_areas=True)
            self._add_object(room_id, treasure_x, treasure_y, ObjectType.TREASURE)
        
        # Place obstacles in random rooms (6 obstacles)
        for _ in range(6):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            obstacle_x, obstacle_y = self._get_random_empty_position_in_room(room_id, avoid_door_areas=True)
            self._add_object(room_id, obstacle_x, obstacle_y, ObjectType.OBSTACLE)
        
        # Place food in random rooms (4 food items)
        for _ in range(4):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            food_x, food_y = self._get_random_empty_position_in_room(room_id, avoid_door_areas=True)
            self._add_object(room_id, food_x, food_y, ObjectType.FOOD)
        
        # Place weapons in random rooms (2 weapons)
        for _ in range(2):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            weapon_x, weapon_y = self._get_random_empty_position_in_room(room_id, avoid_door_areas=True)
            self._add_object(room_id, weapon_x, weapon_y, ObjectType.WEAPON)
    
    def _add_object(self, room_id, x, y, obj_type):
        """Add an object to a room and update the grid"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = obj_type.value
            self.objects.append((room_id, x, y, obj_type))
            self.rooms[room_id]['objects'].append((x, y, obj_type))
    
    def _get_predicates_for_room(self, room_id):
        """Get predicate information for a specific room as first-order logic strings"""
        # Get all objects in the room
        room_objects = self.get_room_objects(room_id)
        
        # Create predicate strings
        predicate_strings = []
        
        # Add AgentAt predicate
        predicate_strings.append(f"AgentAt({room_id})")
        
        # Add At predicates for each object in the room
        object_types_in_room = set()
        for obj in room_objects:
            obj_type = obj[2]
            if obj_type != ObjectType.EMPTY and obj_type != ObjectType.AGENT:
                object_types_in_room.add(obj_type)
        
        for obj_type in object_types_in_room:
            predicate_strings.append(f"At({obj_type.name},{room_id})")
        
        return predicate_strings
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize empty grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Initialize rooms
        self._initialize_rooms()
        
        # Place doors between rooms
        self._place_doors_between_rooms()
        
        # Place other objects randomly
        self._place_objects()
        
        # Place agent in a random empty position in the first room
        first_room_id = "room_0_0"
        agent_x, agent_y = self._get_random_empty_position_in_room(first_room_id, avoid_door_areas=True)
        self.agent_pos = np.array([agent_x, agent_y], dtype=np.int32)
        self.grid[self.agent_pos[1], self.agent_pos[0]] = ObjectType.AGENT.value
        
        # Reset inventory
        self.inventory = ObjectType.EMPTY
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation with first-order predicate information"""
        current_room = self._get_current_room()
        
        # Get predicate strings for current room
        predicate_strings = self._get_predicates_for_room(current_room)
        
        # Join predicates into a single string
        predicates_str = ";".join(predicate_strings)
        
        return {
            'agent_pos': self.agent_pos.copy(),
            'inventory': np.array([self.inventory.value], dtype=np.int32),
            'predicates': predicates_str
        }
    
    def _get_current_room(self):
        """Get the room where the agent is currently located"""
        return self._get_room_id(self.agent_pos[0], self.agent_pos[1])
    
    def _is_valid_move(self, from_pos, to_pos):
        """Check if a move is valid"""
        # Check bounds
        if not (0 <= to_pos[0] < self.width and 0 <= to_pos[1] < self.height):
            return False
        
        # Check if moving into an obstacle
        if self.grid[to_pos[1], to_pos[0]] == ObjectType.OBSTACLE.value:
            return False
        
        # Check if moving between rooms without a door
        from_room = self._get_room_id(from_pos[0], from_pos[1])
        to_room = self._get_room_id(to_pos[0], to_pos[1])
        
        if from_room != to_room:
            # Check if there's a door allowing this transition
            if not self._is_door_between(from_pos, to_pos):
                return False
        
        return True
    
    def step(self, action):
        """Execute one time step"""
        reward = 0
        done = False
        info = {
            'current_room': self._get_current_room(),
            'event': 'moved',
            'inventory': self.inventory
        }
        
        # Save old position and room
        old_pos = self.agent_pos.copy()
        old_room = self._get_current_room()
        
        if action < 4:  # Movement actions
            # Calculate new position based on action
            new_pos = self.agent_pos.copy()
            if action == 0:  # up
                new_pos[1] = max(0, new_pos[1] - 1)
            elif action == 1:  # right
                new_pos[0] = min(self.width - 1, new_pos[0] + 1)
            elif action == 2:  # down
                new_pos[1] = min(self.height - 1, new_pos[1] + 1)
            elif action == 3:  # left
                new_pos[0] = max(0, new_pos[0] - 1)
            
            # Check if move is valid
            if self._is_valid_move(self.agent_pos, new_pos):
                # Update agent position
                self.agent_pos = new_pos
                
                # Update grid - clear old position
                self.grid[old_pos[1], old_pos[0]] = ObjectType.EMPTY.value
                
                # Check what's at the new position
                current_cell = self.grid[self.agent_pos[1], self.agent_pos[0]]
                
                # Check if we changed rooms
                new_room = self._get_current_room()
                if old_room != new_room:
                    info['event'] = 'changed_room'
                    reward = 0.2  # Small reward for exploring new room
                    print(f"Entered {new_room} from {old_room}")
                
                # Update agent position on grid
                self.grid[self.agent_pos[1], self.agent_pos[0]] = ObjectType.AGENT.value
                
                reward = max(reward - 0.01, -0.01)  # Small penalty for each move
            else:
                # Invalid move (hit obstacle, wall, or no door)
                reward = -0.1
                if old_room != self._get_room_id(new_pos[0], new_pos[1]):
                    info['event'] = 'blocked_door'
                    print("Blocked: No door between rooms!")
                else:
                    info['event'] = 'blocked'
        
        elif action == 4:  # Pickup action
            # Check if there's an object at the current position that can be picked up
            current_cell = self.grid[self.agent_pos[1], self.agent_pos[0]]
            
            if current_cell in [ObjectType.KEY.value, ObjectType.TREASURE.value, 
                              ObjectType.FOOD.value, ObjectType.WEAPON.value]:
                
                if self.inventory == ObjectType.EMPTY:
                    # Pick up the object
                    obj_type = ObjectType(current_cell)
                    self.inventory = obj_type
                    
                    # Remove object from grid
                    self.grid[self.agent_pos[1], self.agent_pos[0]] = ObjectType.AGENT.value
                    
                    # Remove from objects list
                    self.objects = [(room_id, x, y, obj) for room_id, x, y, obj in self.objects 
                                  if not (x == self.agent_pos[0] and y == self.agent_pos[1])]
                    
                    # Remove from room objects
                    current_room = self._get_current_room()
                    self.rooms[current_room]['objects'] = [
                        (x, y, obj) for x, y, obj in self.rooms[current_room]['objects']
                        if not (x == self.agent_pos[0] and y == self.agent_pos[1])
                    ]
                    
                    info['event'] = 'picked_up'
                    reward = 0.5
                    print(f"Picked up {obj_type.name.lower()}!")
                else:
                    info['event'] = 'inventory_full'
                    reward = -0.1
                    print("Inventory full! Drop current item first.")
            else:
                info['event'] = 'nothing_to_pickup'
                reward = -0.05
                print("Nothing to pickup here.")
        
        elif action == 5:  # Drop action
            if self.inventory != ObjectType.EMPTY:
                # Check if current cell is empty (except for agent)
                if self.grid[self.agent_pos[1], self.agent_pos[0]] == ObjectType.AGENT.value:
                    # Place object in current position
                    self.grid[self.agent_pos[1], self.agent_pos[0]] = self.inventory.value
                    
                    # Add to objects list
                    current_room = self._get_current_room()
                    self.objects.append((current_room, self.agent_pos[0], self.agent_pos[1], self.inventory))
                    self.rooms[current_room]['objects'].append((self.agent_pos[0], self.agent_pos[1], self.inventory))
                    
                    print(f"Dropped {self.inventory.name.lower()} in {current_room}!")
                    self.inventory = ObjectType.EMPTY
                    info['event'] = 'dropped'
                    reward = 0.2
                else:
                    info['event'] = 'cell_occupied'
                    reward = -0.1
                    print("Cannot drop here - cell is occupied!")
            else:
                info['event'] = 'nothing_to_drop'
                reward = -0.05
                print("Nothing to drop!")
        
        # Check for rewards based on current cell (only if not holding the object)
        current_cell = self.grid[self.agent_pos[1], self.agent_pos[0]]
        if self.inventory == ObjectType.EMPTY:
            if current_cell == ObjectType.KEY.value:
                reward = 2
                info['event'] = 'found_key'
                print("Found a key! +2 reward")
            elif current_cell == ObjectType.TREASURE.value:
                reward = 10
                done = True
                info['event'] = 'found_treasure'
                print("Found the treasure! +10 reward. Episode completed!")
            elif current_cell == ObjectType.FOOD.value:
                reward = 1
                info['event'] = 'found_food'
                print("Found food! +1 reward")
            elif current_cell == ObjectType.WEAPON.value:
                reward = 1.5
                info['event'] = 'found_weapon'
                print("Found a weapon! +1.5 reward")
        
        return self._get_observation(), reward, done, info
    
    def _render_ascii(self):
        """Render the environment as ASCII with improved legibility"""
        # Character mapping for better legibility
        chars = {
            ObjectType.EMPTY.value: '·',
            ObjectType.KEY.value: 'K',
            ObjectType.TREASURE.value: 'T',
            ObjectType.AGENT.value: 'A',
            ObjectType.OBSTACLE.value: 'X',
            ObjectType.FOOD.value: 'F',
            ObjectType.WEAPON.value: 'W'
        }
        
        result = []
        
        # Header with room info
        result.append("=" * 80)
        result.append(f"GridWorld - {self.room_rows}x{self.room_cols} Rooms (Size: {self.room_size}x{self.room_size})")
        result.append(f"Current Room: {self._get_current_room()}")
        result.append(f"Inventory: {self.inventory.name if self.inventory != ObjectType.EMPTY else 'Empty'}")
        
        # Add predicate information
        obs = self._get_observation()
        predicates = obs['predicates'].split(';')
        
        result.append("Predicates:")
        for pred in predicates:
            result.append(f"  {pred}")
        
        result.append("=" * 80)
        result.append("")
        
        # Get current room bounds to highlight it
        current_room = self._get_current_room()
        room_x1, room_y1, room_x2, room_y2 = self._get_room_bounds(current_room)
        
        # Create ASCII representation with room boundaries
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                cell_value = self.grid[y, x]
                
                # Add room boundaries
                if x % self.room_size == 0 and x > 0:
                    # Check if there's a door at this vertical boundary
                    has_door = any(dx == 1 and dy == 0 for dx, dy in self.doors.get((x-1, y), set()))
                    row_str += "|" if not has_door else " "
                
                # Highlight current room with brackets
                if (room_x1 <= x < room_x2 and room_y1 <= y < room_y2):
                    row_str += f"[{chars.get(cell_value, '?')}]"
                else:
                    row_str += f" {chars.get(cell_value, '?')} "
            
            result.append(row_str)
            
            # Add horizontal room boundaries
            if y < self.height - 1 and (y + 1) % self.room_size == 0:
                boundary_line = ""
                for x in range(self.width):
                    if x % self.room_size == 0 and x > 0:
                        boundary_line += "+"
                    
                    # Check if there's a door at this horizontal boundary
                    has_door = any(dx == 0 and dy == 1 for dx, dy in self.doors.get((x, y), set()))
                    boundary_line += "---" if not has_door else "   "
                
                result.append(boundary_line)
        
        result.append("")
        result.append("-" * 80)
        result.append("Legend: A=Agent, K=Key, T=Treasure, X=Obstacle, F=Food, W=Weapon, ·=Empty")
        result.append(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        result.append(f"Current Room Objects: {len(self.get_room_objects(current_room))}")
        result.append("Actions: 0=Up, 1=Right, 2=Down, 3=Left, 4=Pickup, 5=Drop")
        result.append("-" * 80)
        
        return "\n".join(result)
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'ansi':
            return self._render_ascii()
        
        # Original image rendering for 'human' and 'rgb_array' modes
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Create color map for objects
        colors = {
            ObjectType.EMPTY.value: 'white',
            ObjectType.KEY.value: 'yellow',
            ObjectType.TREASURE.value: 'gold',
            ObjectType.AGENT.value: 'blue',
            ObjectType.OBSTACLE.value: 'gray',
            ObjectType.FOOD.value: 'green',
            ObjectType.WEAPON.value: 'red'
        }
        
        # Letter mapping for image rendering
        labels = {
            ObjectType.EMPTY.value: '',
            ObjectType.KEY.value: 'K',
            ObjectType.TREASURE.value: 'T',
            ObjectType.AGENT.value: 'A',
            ObjectType.OBSTACLE.value: 'X',
            ObjectType.FOOD.value: 'F',
            ObjectType.WEAPON.value: 'W'
        }
        
        # Draw grid cells
        for y in range(self.height):
            for x in range(self.width):
                color = colors.get(self.grid[y, x], 'white')
                rect = patches.Rectangle((x, self.height - y - 1), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add letter labels
                label = labels.get(self.grid[y, x], '')
                if label:
                    ax.text(x + 0.5, self.height - y - 0.5, label, 
                           ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw room boundaries
        for x in range(1, self.num_rooms_x):
            line_x = x * self.room_size
            ax.plot([line_x, line_x], [0, self.height], 'r-', linewidth=2, alpha=0.5)
        
        for y in range(1, self.num_rooms_y):
            line_y = y * self.room_size
            ax.plot([0, self.width], [line_y, line_y], 'r-', linewidth=2, alpha=0.5)
        
        # Draw doors as green arcs between cells
        for door_info in self.door_positions:
            x, y, dx, dy = door_info
            
            # Convert to render coordinates
            render_x = x + 0.5
            render_y = self.height - y - 0.5
            
            if dx == 1 and dy == 0:  # Right door
                # Draw door on the right side of the cell
                ax.plot([render_x + 0.5, render_x + 0.5], 
                        [render_y - 0.3, render_y + 0.3], 
                        'g-', linewidth=4, alpha=0.8)
            elif dx == -1 and dy == 0:  # Left door
                # Draw door on the left side of the cell
                ax.plot([render_x - 0.5, render_x - 0.5], 
                        [render_y - 0.3, render_y + 0.3], 
                        'g-', linewidth=4, alpha=0.8)
            elif dx == 0 and dy == 1:  # Down door
                # Draw door on the bottom side of the cell
                ax.plot([render_x - 0.3, render_x + 0.3], 
                        [render_y - 0.5, render_y - 0.5], 
                        'g-', linewidth=4, alpha=0.8)
            elif dx == 0 and dy == -1:  # Up door
                # Draw door on the top side of the cell
                ax.plot([render_x - 0.3, render_x + 0.3], 
                        [render_y + 0.5, render_y + 0.5], 
                        'g-', linewidth=4, alpha=0.8)
        
        # Add room labels
        for room_id, room_info in self.rooms.items():
            x1, y1, x2, y2 = room_info['bounds']
            center_x = (x1 + x2) / 2
            center_y = self.height - (y1 + y2) / 2
            ax.text(center_x, center_y, room_id, ha='center', va='center', 
                   fontsize=6, color='red', alpha=0.7, weight='bold')
        
        # Add inventory and predicate info to the plot
        obs = self._get_observation()
        predicates = obs['predicates'].split(';')
        
        info_text = f"Inventory: {self.inventory.name if self.inventory != ObjectType.EMPTY else 'Empty'}\n"
        info_text += "Predicates:\n"
        for pred in predicates:
            info_text += f"  {pred}\n"
        
        ax.text(0.5, self.height + 3.0, info_text, 
               ha='left', va='center', fontsize=10, weight='bold', transform=ax.transData,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height + 5)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, self.width + 1))
        ax.set_yticks(np.arange(0, self.height + 1))
        ax.grid(True, color='black', linewidth=0.5)
        ax.set_title(f'GridWorld - {self.room_rows}x{self.room_cols} Rooms (First-Order Predicate Observations)')
        
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
    
    def close(self):
        """Clean up resources"""
        plt.close('all')
    
    def get_room_objects(self, room_id):
        """Get all objects in a specific room"""
        return self.rooms.get(room_id, {}).get('objects', [])
    
    def get_agent_room(self):
        """Get the current room of the agent"""
        return self._get_current_room()
    
    def get_room_connections(self, room_id):
        """Get all rooms connected to the given room by doors"""
        connected_rooms = set()
        x_start, y_start, x_end, y_end = self._get_room_bounds(room_id)
        
        # Check all boundary cells for doors
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                for dx, dy in self.doors.get((x, y), set()):
                    # Calculate the room we'd enter through this door
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        connected_rooms.add(self._get_room_id(new_x, new_y))
        
        return connected_rooms
    
    def is_fully_connected(self):
        """Check if all rooms are reachable from room_0_0"""
        visited = set()
        stack = ["room_0_0"]
        
        while stack:
            current_room = stack.pop()
            if current_room in visited:
                continue
            visited.add(current_room)
            
            # Add all connected rooms to the stack
            connected_rooms = self.get_room_connections(current_room)
            stack.extend(connected_rooms - visited)
        
        return len(visited) == len(self.rooms)

# Example usage and testing
if __name__ == "__main__":
    # Create a 3x3 grid of rooms
    env = GridWorldEnv(room_size=5, room_rows=3, room_cols=3)
    
    # Test the environment
    obs = env.reset()
    print(f"Initial room: {env.get_agent_room()}")
    
    # Show the new observation structure
    print("\nInitial observation:")
    print(f"Agent position: {obs['agent_pos']}")
    print(f"Inventory: {obs['inventory']}")
    print(f"Predicates: {obs['predicates']}")
    
    # Check if all rooms are reachable
    print(f"\nAll rooms are reachable: {env.is_fully_connected()}")
    
    # Show initial configuration with image
    print("\nInitial configuration (image):")
    env.render(mode='human')
    
    # Show ASCII representation
    print("\nASCII representation:")
    print(env.render(mode='ansi'))
    
    # Take some actions to demonstrate the new observation system
    actions = [1, 1, 2, 4]  # Right, Right, Down, Pickup
    
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        print(f"\nStep {i}: Action={action}, Reward={reward:.2f}, Room={info['current_room']}, Event={info['event']}")
        
        # Show the new observation with predicates
        print(f"Observation - Predicates: {obs['predicates']}")
        print(env.render(mode='ansi'))
        
        if done:
            print("Episode finished!")
            break