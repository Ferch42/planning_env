import numpy as np
import random
from enum import Enum

class ObjectType(Enum):
    EMPTY = 0
    KEY = 1
    TREASURE = 2
    FOOD = 3
    TOOL = 4

class GridWorld:
    def __init__(self, num_rooms=25, room_size=5):
        self.num_rooms = num_rooms
        self.room_size = room_size
        self.rooms_per_side = int(np.sqrt(num_rooms))
        
        # Account for walls between rooms
        self.grid_size = self.rooms_per_side * room_size + (self.rooms_per_side - 1)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Store table positions and room IDs
        self.table_positions = {}
        self.room_ids = {}  # Maps position to room ID
        
        # Initialize agent at position (1,1)
        self.agent_pos = (1, 1)
        self.agent_inventory = None
        
        # Precompute important transitions
        self.door_transitions = []
        self.object_transitions = []
        
        self._build_rooms()
        self._assign_objects()
        self._assign_room_ids()
        self._precompute_transitions()
    
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
        """Assign objects to room centers - ensure at least one of each type"""
        # Get all object types except EMPTY
        object_types = [obj.value for obj in ObjectType if obj != ObjectType.EMPTY]
        
        # Get all room centers
        room_centers = []
        for room_x in range(self.rooms_per_side):
            for room_y in range(self.rooms_per_side):
                center_x = room_x * (self.room_size + 1) + self.room_size // 2
                center_y = room_y * (self.room_size + 1) + self.room_size // 2
                self.table_positions[(center_x, center_y)] = (room_x, room_y)
                room_centers.append((center_x, center_y))
        
        # Shuffle room centers to assign objects randomly
        random.shuffle(room_centers)
        
        # First, ensure at least one of each object type
        for i, obj_type in enumerate(object_types):
            if i < len(room_centers):
                center_x, center_y = room_centers[i]
                self.grid[center_x, center_y] = obj_type
        
        # Then, assign random objects to remaining rooms (80% chance)
        for i in range(len(object_types), len(room_centers)):
            if random.random() > 0.2:  # 80% chance for remaining rooms
                center_x, center_y = room_centers[i]
                obj_type = random.choice(object_types)
                self.grid[center_x, center_y] = obj_type
    
    def _assign_room_ids(self):
        """Assign room IDs to all positions in the grid"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Calculate which room this position belongs to
                room_x = x // (self.room_size + 1)
                room_y = y // (self.room_size + 1)
                room_id = room_x + room_y * self.rooms_per_side
                self.room_ids[(x, y)] = room_id
    
    def _precompute_transitions(self):
        """Precompute all possible door and object transitions"""
        # Precompute door transitions - FIXED VERSION
        door_pos = self.room_size // 2
        
        # Horizontal doors (vertical walls)
        for i in range(1, self.rooms_per_side):
            wall_row = i * (self.room_size + 1) - 1
            for j in range(self.rooms_per_side):
                door_col = j * (self.room_size + 1) + door_pos
                
                # The door is at (wall_row, door_col)
                # Moving through the door is a single step from one side to the other
                
                # From above the door to the door position (DOWN action)
                self.door_transitions.append({
                    'prev_position': (wall_row - 1, door_col),
                    'action': 1,  # DOWN
                    'next_position': (wall_row, door_col),
                    'type': 'door'
                })
                
                # From the door position to below the door (DOWN action)
                self.door_transitions.append({
                    'prev_position': (wall_row, door_col),
                    'action': 1,  # DOWN
                    'next_position': (wall_row + 1, door_col),
                    'type': 'door'
                })
                
                # From below the door to the door position (UP action)
                self.door_transitions.append({
                    'prev_position': (wall_row + 1, door_col),
                    'action': 0,  # UP
                    'next_position': (wall_row, door_col),
                    'type': 'door'
                })
                
                # From the door position to above the door (UP action)
                self.door_transitions.append({
                    'prev_position': (wall_row, door_col),
                    'action': 0,  # UP
                    'next_position': (wall_row - 1, door_col),
                    'type': 'door'
                })
        
        # Vertical doors (horizontal walls)
        for i in range(1, self.rooms_per_side):
            wall_col = i * (self.room_size + 1) - 1
            for j in range(self.rooms_per_side):
                door_row = j * (self.room_size + 1) + door_pos
                
                # From left of the door to the door position (RIGHT action)
                self.door_transitions.append({
                    'prev_position': (door_row, wall_col - 1),
                    'action': 3,  # RIGHT
                    'next_position': (door_row, wall_col),
                    'type': 'door'
                })
                
                # From the door position to right of the door (RIGHT action)
                self.door_transitions.append({
                    'prev_position': (door_row, wall_col),
                    'action': 3,  # RIGHT
                    'next_position': (door_row, wall_col + 1),
                    'type': 'door'
                })
                
                # From right of the door to the door position (LEFT action)
                self.door_transitions.append({
                    'prev_position': (door_row, wall_col + 1),
                    'action': 2,  # LEFT
                    'next_position': (door_row, wall_col),
                    'type': 'door'
                })
                
                # From the door position to left of the door (LEFT action)
                self.door_transitions.append({
                    'prev_position': (door_row, wall_col),
                    'action': 2,  # LEFT
                    'next_position': (door_row, wall_col - 1),
                    'type': 'door'
                })
        
        # Precompute object transitions (all table positions)
        for table_pos in self.table_positions.keys():
            self.object_transitions.append({
                'prev_position': table_pos,
                'action': 4,  # TOGGLE
                'next_position': table_pos,
                'type': 'object'
            })
    
    def get_current_room_id(self):
        """Get the ID of the room the agent is currently in"""
        return self.room_ids.get(self.agent_pos, -1)
    
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
    
    def get_important_transitions(self):
        """Get the precomputed important transitions"""
        return {
            'door_transitions': self.door_transitions,
            'object_transitions': self.object_transitions
        }
    
    def get_state(self):
        """Return current state including grid, room ID, and inventory"""
        grid_state = self.grid.copy()
        x, y = self.agent_pos
        grid_state[x, y] = -1  # Mark agent position
        
        return {
            'grid': grid_state,
            'room_id': self.get_current_room_id(),
            'inventory': self.agent_inventory
        }
    
    def render(self):
        """Display the current state"""
        state = self.get_state()
        print(state['grid'])
        print(f"Room ID: {state['room_id']}, Inventory: {state['inventory']}")


def test_door_transitions():
    """Test that door transitions correctly move the agent between rooms"""
    print("=== Testing Door Transitions ===")
    
    # Create a smaller grid for easier testing
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Get door transitions
    transitions = world.get_important_transitions()
    door_trans = transitions['door_transitions']
    
    print(f"Found {len(door_trans)} door transitions")
    
    # Test a few door transitions
    test_cases = door_trans[:4]  # Test first 4 door transitions
    
    for i, trans in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Transition: {trans['prev_position']} -> {trans['next_position']} via action {trans['action']}")
        
        # Set agent to starting position
        world.agent_pos = trans['prev_position']
        start_room = world.get_current_room_id()
        print(f"  Start position: {world.agent_pos}, Room: {start_room}")
        
        # Execute the action
        world.step(trans['action'])
        end_room = world.get_current_room_id()
        print(f"  End position: {world.agent_pos}, Room: {end_room}")
        
        # Verify the transition worked
        expected_pos = trans['next_position']
        if world.agent_pos == expected_pos:
            print("  ✓ SUCCESS: Agent moved to expected position")
        else:
            print(f"  ✗ FAILURE: Expected {expected_pos}, got {world.agent_pos}")
        
        if start_room != end_room:
            print("  ✓ SUCCESS: Agent changed rooms")
        else:
            # Check if this was supposed to be a room change
            prev_room = world.room_ids[trans['prev_position']]
            next_room = world.room_ids[trans['next_position']]
            if prev_room != next_room:
                print("  ✗ FAILURE: Agent should have changed rooms but didn't")
            else:
                print("  ✓ SUCCESS: No room change expected")


def test_object_transitions():
    """Test that object transitions correctly pick up and put down objects"""
    print("\n=== Testing Object Transitions ===")
    
    # Create a smaller grid for easier testing
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Get object transitions
    transitions = world.get_important_transitions()
    object_trans = transitions['object_transitions']
    
    print(f"Found {len(object_trans)} object transitions")
    
    # Test picking up an object
    print("\nTest 1: Picking up an object")
    # Find a table with an object
    table_with_object = None
    for table_pos in world.table_positions.keys():
        if world.grid[table_pos] >= 2:  # Has an object
            table_with_object = table_pos
            break
    
    if table_with_object:
        print(f"  Testing with table at {table_with_object} with object {world.grid[table_with_object]}")
        
        # Set agent to table position
        world.agent_pos = table_with_object
        world.agent_inventory = None
        
        # Execute toggle action
        world.step(4)  # TOGGLE action
        
        # Check if object was picked up
        if world.agent_inventory is not None and world.grid[table_with_object] == 0:
            print(f"  ✓ SUCCESS: Object {world.agent_inventory} picked up from table")
        else:
            print(f"  ✗ FAILURE: Object not picked up. Inventory: {world.agent_inventory}, Table: {world.grid[table_with_object]}")
    else:
        print("  ✗ SKIPPED: No table with object found")
    
    # Test putting down an object
    print("\nTest 2: Putting down an object")
    # Find an empty table
    empty_table = None
    for table_pos in world.table_positions.keys():
        if world.grid[table_pos] == 0:  # Empty table
            empty_table = table_pos
            break
    
    if empty_table and world.agent_inventory is not None:
        print(f"  Testing with empty table at {empty_table}, agent has object {world.agent_inventory}")
        
        # Set agent to empty table position
        world.agent_pos = empty_table
        
        # Execute toggle action
        world.step(4)  # TOGGLE action
        
        # Check if object was put down
        if world.agent_inventory is None and world.grid[empty_table] >= 2:
            print(f"  ✓ SUCCESS: Object {world.grid[empty_table]} put down on table")
        else:
            print(f"  ✗ FAILURE: Object not put down. Inventory: {world.agent_inventory}, Table: {world.grid[empty_table]}")
    else:
        print("  ✗ SKIPPED: No empty table found or agent has no object")


def test_invalid_actions():
    """Test that invalid actions don't change the agent's state"""
    print("\n=== Testing Invalid Actions ===")
    
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Test moving into a wall
    print("Test 1: Moving into a wall")
    # Find a wall position next to the agent
    world.agent_pos = (0, 1)  # Top row
    start_pos = world.agent_pos
    print(f"  Start position: {start_pos}")
    
    # Try to move up (should fail)
    world.step(0)  # UP
    
    if world.agent_pos == start_pos:
        print("  ✓ SUCCESS: Agent did not move into wall")
    else:
        print(f"  ✗ FAILURE: Agent moved to {world.agent_pos}")
    
    # Test picking up object when not at table
    print("\nTest 2: Picking up object when not at table")
    # Find a non-table position
    non_table_pos = (1, 1)  # Should be a floor position
    world.agent_pos = non_table_pos
    world.agent_inventory = None
    start_inventory = world.agent_inventory
    
    # Try to pick up (should fail)
    world.step(4)  # TOGGLE
    
    if world.agent_inventory == start_inventory:
        print("  ✓ SUCCESS: Inventory unchanged when not at table")
    else:
        print(f"  ✗ FAILURE: Inventory changed to {world.agent_inventory}")


if __name__ == "__main__":
    # Run all tests
    test_door_transitions()
    test_object_transitions()
    test_invalid_actions()
    
    # Show final state of the test world
    print("\n=== Final Test World State ===")
    world = GridWorld(num_rooms=4, room_size=3)
    world.render()
    
    # Show some precomputed transitions
    transitions = world.get_important_transitions()
    print(f"\nSample Door Transitions:")
    for trans in transitions['door_transitions'][:3]:
        print(f"  {trans}")
    
    print(f"\nSample Object Transitions:")
    for trans in transitions['object_transitions'][:3]:
        print(f"  {trans}")