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
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, width=25, height=25, room_size=5, room_rows=5, room_cols=5):
        super(GridWorldEnv, self).__init__()
        
        self.room_size = room_size
        self.room_rows = room_rows
        self.room_cols = room_cols
        self.width = room_cols * room_size
        self.height = room_rows * room_size
        
        self.num_rooms_x = room_cols
        self.num_rooms_y = room_rows
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid of objects + agent position
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=len(ObjectType)-1, 
                             shape=(self.height, self.width), dtype=np.int32),
            'agent_pos': spaces.Box(low=0, high=max(self.width, self.height), 
                                  shape=(2,), dtype=np.int32)
        })
        
        # Initialize environment state
        self.grid = None
        self.agent_pos = None
        self.rooms = {}
        self.objects = []  # List of (room_id, x, y, object_type)
        self.doors = defaultdict(set)  # Maps (x,y) to set of allowed directions
        self.door_positions = set()  # Store door positions for rendering
        
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
    
    def _get_random_empty_position_in_room(self, room_id):
        """Get a random empty position within a room"""
        x_start, y_start, x_end, y_end = self._get_room_bounds(room_id)
        
        # Try to find an empty position
        for _ in range(100):
            x = np.random.randint(x_start, x_end)
            y = np.random.randint(y_start, y_end)
            if self.grid[y, x] == ObjectType.EMPTY.value:
                return x, y
        
        # If no empty spot found, return any non-occupied position
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if self.grid[y, x] == ObjectType.EMPTY.value:
                    return x, y
        
        # Last resort: center of room
        return (x_start + x_end) // 2, (y_start + y_end) // 2
    
    def _place_doors_between_rooms(self):
        """Place doors as transitions between rooms - only 0-1 door per boundary"""
        self.doors.clear()
        self.door_positions.clear()
        
        # Place vertical doors (between rooms horizontally)
        for room_x in range(self.num_rooms_x - 1):
            for room_y in range(self.num_rooms_y):
                # 50% chance to place a door on this boundary
                if np.random.random() < 0.5:
                    # Choose 1 random vertical position for door in this boundary
                    y = np.random.randint(room_y * self.room_size, (room_y + 1) * self.room_size)
                    x = (room_x + 1) * self.room_size
                    
                    if y < self.height:
                        # Add door in both directions
                        self.doors[(x-1, y)].add((1, 0))  # Right direction
                        self.doors[(x, y)].add((-1, 0))   # Left direction
                        self.door_positions.add((x-1, y, 1, 0))  # Store for rendering
                        self.door_positions.add((x, y, -1, 0))   # Store for rendering
        
        # Place horizontal doors (between rooms vertically)
        for room_x in range(self.num_rooms_x):
            for room_y in range(self.num_rooms_y - 1):
                # 50% chance to place a door on this boundary
                if np.random.random() < 0.5:
                    # Choose 1 random horizontal position for door in this boundary
                    x = np.random.randint(room_x * self.room_size, (room_x + 1) * self.room_size)
                    y = (room_y + 1) * self.room_size
                    
                    if x < self.width:
                        # Add door in both directions
                        self.doors[(x, y-1)].add((0, 1))  # Down direction
                        self.doors[(x, y)].add((0, -1))   # Up direction
                        self.door_positions.add((x, y-1, 0, 1))  # Store for rendering
                        self.door_positions.add((x, y, 0, -1))   # Store for rendering
    
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
        """Place objects randomly in different rooms - fewer objects"""
        self.objects = []
        
        # Place keys in random rooms (3 keys)
        for _ in range(3):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            key_x, key_y = self._get_random_empty_position_in_room(room_id)
            self._add_object(room_id, key_x, key_y, ObjectType.KEY)
        
        # Place treasures in random rooms (2 treasures)
        for _ in range(2):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            treasure_x, treasure_y = self._get_random_empty_position_in_room(room_id)
            self._add_object(room_id, treasure_x, treasure_y, ObjectType.TREASURE)
        
        # Place obstacles in random rooms (6 obstacles)
        for _ in range(6):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            obstacle_x, obstacle_y = self._get_random_empty_position_in_room(room_id)
            self._add_object(room_id, obstacle_x, obstacle_y, ObjectType.OBSTACLE)
        
        # Place food in random rooms (4 food items)
        for _ in range(4):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            food_x, food_y = self._get_random_empty_position_in_room(room_id)
            self._add_object(room_id, food_x, food_y, ObjectType.FOOD)
        
        # Place weapons in random rooms (2 weapons)
        for _ in range(2):
            room_x = np.random.randint(0, self.num_rooms_x)
            room_y = np.random.randint(0, self.num_rooms_y)
            room_id = f"room_{room_x}_{room_y}"
            weapon_x, weapon_y = self._get_random_empty_position_in_room(room_id)
            self._add_object(room_id, weapon_x, weapon_y, ObjectType.WEAPON)
    
    def _add_object(self, room_id, x, y, obj_type):
        """Add an object to a room and update the grid"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = obj_type.value
            self.objects.append((room_id, x, y, obj_type))
            self.rooms[room_id]['objects'].append((x, y, obj_type))
    
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
        agent_x, agent_y = self._get_random_empty_position_in_room(first_room_id)
        self.agent_pos = np.array([agent_x, agent_y], dtype=np.int32)
        self.grid[self.agent_pos[1], self.agent_pos[0]] = ObjectType.AGENT.value
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos.copy()
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
            'event': 'moved'
        }
        
        # Save old position and room
        old_pos = self.agent_pos.copy()
        old_room = self._get_current_room()
        
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
            else:
                reward = max(reward - 0.01, -0.01)  # Small penalty for each move
            
            # Update agent position on grid
            self.grid[self.agent_pos[1], self.agent_pos[0]] = ObjectType.AGENT.value
        else:
            # Invalid move (hit obstacle, wall, or no door)
            reward = -0.1
            if old_room != self._get_room_id(new_pos[0], new_pos[1]):
                info['event'] = 'blocked_door'
                print("Blocked: No door between rooms!")
            else:
                info['event'] = 'blocked'
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the environment with letters"""
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
        
        # Draw grid cells
        for y in range(self.height):
            for x in range(self.width):
                color = colors.get(self.grid[y, x], 'white')
                rect = patches.Rectangle((x, self.height - y - 1), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add object labels
                if self.grid[y, x] == ObjectType.KEY.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'K', 
                           ha='center', va='center', fontweight='bold')
                elif self.grid[y, x] == ObjectType.TREASURE.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'T', 
                           ha='center', va='center', fontweight='bold')
                elif self.grid[y, x] == ObjectType.AGENT.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'A', 
                           ha='center', va='center', fontweight='bold')
                elif self.grid[y, x] == ObjectType.OBSTACLE.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'X', 
                           ha='center', va='center', fontweight='bold')
                elif self.grid[y, x] == ObjectType.FOOD.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'F', 
                           ha='center', va='center', fontweight='bold')
                elif self.grid[y, x] == ObjectType.WEAPON.value:
                    ax.text(x + 0.5, self.height - y - 0.5, 'W', 
                           ha='center', va='center', fontweight='bold')
        
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
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, self.width + 1))
        ax.set_yticks(np.arange(0, self.height + 1))
        ax.grid(True, color='black', linewidth=0.5)
        ax.set_title(f'GridWorld Environment - {self.room_rows}x{self.room_cols} Rooms (Total: {self.room_rows * self.room_cols})')
        
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

# Example usage and testing
if __name__ == "__main__":
    # Create a 5x5 grid of rooms (25 total rooms)
    env = GridWorldEnv(room_size=5, room_rows=5, room_cols=5)
    
    # Test the environment
    obs = env.reset()
    print(f"Initial room: {env.get_agent_room()}")
    print(f"Objects in initial room: {env.get_room_objects(env.get_agent_room())}")
    
    # Print room connections
    print(f"\nRoom connections for {env.get_agent_room()}: {env.get_room_connections(env.get_agent_room())}")
    
    # Print all rooms and their objects
    print("\nAll rooms and their objects:")
    object_count = 0
    for room_id, room_info in env.rooms.items():
        objects = room_info['objects']
        if objects:
            print(f"{room_id}: {objects}")
            object_count += len(objects)
    
    print(f"\nTotal objects in environment: {object_count}")
    
    # Take some random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Room={info['current_room']}, Event={info['event']}")
        
        if done:
            print("Episode finished!")
            break
    
    # Render the final state
    env.render()