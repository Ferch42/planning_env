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
        # Precompute door transitions - Only transitions that change rooms
        door_pos = self.room_size // 2
        
        # Horizontal doors (vertical walls)
        for i in range(1, self.rooms_per_side):
            wall_row = i * (self.room_size + 1) - 1
            for j in range(self.rooms_per_side):
                door_col = j * (self.room_size + 1) + door_pos
                
                # The door is at (wall_row, door_col)
                # Moving through the door is a single step from one side to the door or from the door to the other side
                
                # From above the door to the door position (DOWN action)
                prev_pos = (wall_row - 1, door_col)
                next_pos = (wall_row, door_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 1,  # DOWN
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From the door position to below the door (DOWN action)
                prev_pos = (wall_row, door_col)
                next_pos = (wall_row + 1, door_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 1,  # DOWN
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From below the door to the door position (UP action)
                prev_pos = (wall_row + 1, door_col)
                next_pos = (wall_row, door_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 0,  # UP
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From the door position to above the door (UP action)
                prev_pos = (wall_row, door_col)
                next_pos = (wall_row - 1, door_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 0,  # UP
                        'next_position': next_pos,
                        'type': 'door'
                    })
        
        # Vertical doors (horizontal walls)
        for i in range(1, self.rooms_per_side):
            wall_col = i * (self.room_size + 1) - 1
            for j in range(self.rooms_per_side):
                door_row = j * (self.room_size + 1) + door_pos
                
                # From left of the door to the door position (RIGHT action)
                prev_pos = (door_row, wall_col - 1)
                next_pos = (door_row, wall_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 3,  # RIGHT
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From the door position to right of the door (RIGHT action)
                prev_pos = (door_row, wall_col)
                next_pos = (door_row, wall_col + 1)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 3,  # RIGHT
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From right of the door to the door position (LEFT action)
                prev_pos = (door_row, wall_col + 1)
                next_pos = (door_row, wall_col)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 2,  # LEFT
                        'next_position': next_pos,
                        'type': 'door'
                    })
                
                # From the door position to left of the door (LEFT action)
                prev_pos = (door_row, wall_col)
                next_pos = (door_row, wall_col - 1)
                if self.room_ids[prev_pos] != self.room_ids[next_pos]:
                    self.door_transitions.append({
                        'prev_position': prev_pos,
                        'action': 2,  # LEFT
                        'next_position': next_pos,
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
                if self.grid[self.agent_pos] >= 1:
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


class Agent:
    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.knowledge_base = {
            'known_rooms': set(),  # Room IDs the agent has visited
            'room_connections': set(),  # Tuples (room1, room2) for connected rooms
            'object_locations': {},  # Maps object_id to room_id where objects are located
            'current_room': None,
            'previous_room': None,  # Track previous room to detect connections
            'previous_inventory': None  # Track inventory changes
        }
        
        # Initialize with starting room knowledge
        self._update_knowledge()
    
    def _update_knowledge(self):
        """Update knowledge based on current state"""
        state = self.grid_world.get_state()
        current_room = state['room_id']
        current_inventory = state['inventory']
        
        # Update room tracking
        self.knowledge_base['previous_room'] = self.knowledge_base['current_room']
        self.knowledge_base['current_room'] = current_room
        
        # Mark current room as known
        self.knowledge_base['known_rooms'].add(current_room)
        
        # Detect and record room connections
        if (self.knowledge_base['previous_room'] is not None and 
            self.knowledge_base['previous_room'] != current_room):
            
            # Add bidirectional connection
            room1 = self.knowledge_base['previous_room']
            room2 = current_room
            connection = tuple(sorted([room1, room2]))  # Sort to avoid duplicates like (0,1) and (1,0)
            self.knowledge_base['room_connections'].add(connection)
        
        # Detect object putdown: when inventory changes from an object to None AND agent is at a table
        if (self.knowledge_base['previous_inventory'] is not None and 
            current_inventory is None and
            self.grid_world.agent_pos in self.grid_world.table_positions):
            
            # Agent just put down an object at a table - record its location
            obj_id = self.knowledge_base['previous_inventory']
            self.knowledge_base['object_locations'][obj_id] = current_room
        
        # Detect object pickup: when inventory changes from None to an object
        elif (self.knowledge_base['previous_inventory'] is None and 
              current_inventory is not None and
              self.grid_world.agent_pos in self.grid_world.table_positions):
            
            # Agent just picked up an object - remove it from known locations
            obj_id = current_inventory
            if obj_id in self.knowledge_base['object_locations']:
                del self.knowledge_base['object_locations'][obj_id]
        
        # Update previous inventory for next comparison
        self.knowledge_base['previous_inventory'] = current_inventory
    
    def step(self, action):
        """Take an action and update knowledge"""
        # Execute the action
        self.grid_world.step(action)
        
        # Update knowledge after action
        self._update_knowledge()
    
    def get_knowledge(self):
        """Return the current knowledge base"""
        return self.knowledge_base.copy()
    
    def knows_room(self, room_id):
        """Check if agent knows about a room"""
        return room_id in self.knowledge_base['known_rooms']
    
    def knows_connection(self, room1, room2):
        """Check if agent knows two rooms are connected"""
        connection = tuple(sorted([room1, room2]))
        return connection in self.knowledge_base['room_connections']
    
    def knows_object_location(self, obj_id):
        """Check if agent knows where an object is located"""
        return obj_id in self.knowledge_base['object_locations']
    
    def get_known_objects(self):
        """Get all objects whose locations are known"""
        return list(self.knowledge_base['object_locations'].keys())
    
    def get_connected_rooms(self, room_id):
        """Get all rooms known to be connected to a given room"""
        connected = set()
        for conn in self.knowledge_base['room_connections']:
            if room_id in conn:
                other_room = conn[0] if conn[1] == room_id else conn[1]
                connected.add(other_room)
        return connected
    
    def get_known_connectivity_graph(self):
        """Return the complete connectivity graph as a dictionary"""
        graph = {}
        for room in self.knowledge_base['known_rooms']:
            graph[room] = self.get_connected_rooms(room)
        return graph
    
    def render_knowledge(self):
        """Display the agent's current knowledge"""
        kb = self.knowledge_base
        
        print("=== Agent Knowledge Base ===")
        print(f"Current Room: {kb['current_room']}")
        print(f"Previous Room: {kb['previous_room']}")
        print(f"Known Rooms: {sorted(kb['known_rooms'])}")
        
        print("\nRoom Connections:")
        if kb['room_connections']:
            for conn in sorted(kb['room_connections']):
                print(f"  Room {conn[0]} ↔ Room {conn[1]}")
        else:
            print("  No connections discovered yet")
        
        print("\nConnectivity Graph:")
        graph = self.get_known_connectivity_graph()
        for room, connected in sorted(graph.items()):
            print(f"  Room {room} → {sorted(connected)}")
        
        print("\nObject Locations (objects on tables):")
        if kb['object_locations']:
            for obj_id, room_id in kb['object_locations'].items():
                obj_name = ObjectType(obj_id).name
                print(f"  {obj_name} is in Room {room_id}")
        else:
            print("  No object locations known")
import unittest

class TestGridWorldAgentComprehensive(unittest.TestCase):
    """Comprehensive test suite for GridWorld and Agent classes with all object types"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.world = GridWorld(num_rooms=9, room_size=5)  # Larger grid for more comprehensive testing
        self.agent = Agent(self.world)
    
    def test_01_initial_state(self):
        """Test initial state of world and agent"""
        print("\n=== Test 01: Initial State ===")
        
        # Test world initial state
        state = self.world.get_state()
        self.assertEqual(state['room_id'], 0)
        self.assertEqual(state['inventory'], None)
        self.assertEqual(self.world.agent_pos, (1, 1))
        
        # Test agent initial knowledge
        kb = self.agent.get_knowledge()
        self.assertEqual(kb['known_rooms'], {0})
        self.assertEqual(kb['room_connections'], set())
        self.assertEqual(kb['object_locations'], {})
        self.assertEqual(kb['current_room'], 0)
        
        print("✓ World and agent initialized correctly")
    
    def test_02_all_object_types_pickup(self):
        """Test that all object types can be picked up and removed from known locations"""
        print("\n=== Test 02: All Object Types Pickup ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 4, "Need at least 4 tables for this test")
        
        # Test all object types
        object_types = [
            ObjectType.KEY,
            ObjectType.TREASURE, 
            ObjectType.FOOD,
            ObjectType.TOOL
        ]
        
        for obj_type in object_types:
            with self.subTest(obj_type=obj_type):
                table_pos = table_positions[object_types.index(obj_type)]
                
                # Place object and pick it up
                self.world.grid[table_pos] = obj_type.value
                self.world.agent_pos = table_pos
                self.world.agent_inventory = None  # Reset inventory
                self.agent._update_knowledge()
                
                # Pick up object
                self.agent.step(4)
                
                # Verify pickup worked
                self.assertEqual(self.world.agent_inventory, obj_type.value)
                self.assertEqual(self.world.grid[table_pos], ObjectType.EMPTY.value)
                
                # Object should NOT be in known locations
                kb = self.agent.get_knowledge()
                self.assertNotIn(obj_type.value, kb['object_locations'])
                
                # Reset for next test
                self.world.agent_inventory = None
                self.agent.knowledge_base['previous_inventory'] = None
        
        print("✓ All object types can be picked up and removed from known locations")
    
    def test_03_object_putdown_all_types(self):
        """Test that putting down all object types records their locations"""
        print("\n=== Test 03: Object Putdown All Types ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 8, "Need at least 8 tables for this test")
        
        object_types = [
            ObjectType.KEY,
            ObjectType.TREASURE,
            ObjectType.FOOD, 
            ObjectType.TOOL
        ]
        
        for obj_type in object_types:
            with self.subTest(obj_type=obj_type):
                # Use two tables per object type
                start_idx = object_types.index(obj_type) * 2
                source_table = table_positions[start_idx]
                dest_table = table_positions[start_idx + 1]
                dest_room = self.world.room_ids[dest_table]
                
                # Setup: object at source table, empty destination
                self.world.grid[source_table] = obj_type.value
                self.world.grid[dest_table] = ObjectType.EMPTY.value
                
                # Pick up object
                self.world.agent_pos = source_table
                self.world.agent_inventory = None
                self.agent._update_knowledge()
                self.agent.step(4)
                
                # Put down at destination
                self.world.agent_pos = dest_table
                self.agent._update_knowledge()
                self.agent.step(4)
                
                # Verify putdown
                kb = self.agent.get_knowledge()
                self.assertIn(obj_type.value, kb['object_locations'])
                self.assertEqual(kb['object_locations'][obj_type.value], dest_room)
                self.assertEqual(self.world.agent_inventory, None)
                self.assertEqual(self.world.grid[dest_table], obj_type.value)
        
        print("✓ All object types put down and locations recorded correctly")
    
    def test_04_complex_object_movement_scenarios(self):
        """Test complex object movement scenarios between multiple rooms"""
        print("\n=== Test 04: Complex Object Movement Scenarios ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 6, "Need at least 6 tables for this test")
        
        # Scenario 1: Chain movement through multiple tables
        tables_chain = table_positions[:3]
        obj_id = ObjectType.KEY.value
        
        print("  Testing chain movement...")
        for i, table_pos in enumerate(tables_chain):
            # Place object if first table, otherwise ensure empty
            if i == 0:
                self.world.grid[table_pos] = obj_id
            else:
                self.world.grid[table_pos] = ObjectType.EMPTY.value
            
            # Move to table and pick up/put down
            self.world.agent_pos = table_pos
            self.agent._update_knowledge()
            
            if i == 0:
                # First table: pick up object
                self.agent.step(4)
                self.assertEqual(self.world.agent_inventory, obj_id)
            else:
                # Subsequent tables: put down and pick up again
                self.agent.step(4)  # Put down
                self.assertEqual(self.world.agent_inventory, None)
                self.assertEqual(self.world.grid[table_pos], obj_id)
                
                # Verify location recorded
                kb = self.agent.get_knowledge()
                self.assertIn(obj_id, kb['object_locations'])
                self.assertEqual(kb['object_locations'][obj_id], self.world.room_ids[table_pos])
                
                # Pick up again for next movement (except last)
                if i < len(tables_chain) - 1:
                    self.agent.step(4)
                    self.assertEqual(self.world.agent_inventory, obj_id)
        
        print("  ✓ Chain movement test passed")
        
        # Scenario 2: Multiple objects in different rooms
        print("  Testing multiple objects...")
        obj1_id, obj2_id = ObjectType.TREASURE.value, ObjectType.FOOD.value
        table1, table2, table3, table4 = table_positions[3:7]
        
        # Setup objects
        self.world.grid[table1] = obj1_id
        self.world.grid[table2] = obj2_id
        self.world.grid[table3] = ObjectType.EMPTY.value
        self.world.grid[table4] = ObjectType.EMPTY.value
        
        # Move obj1 to table3
        self.world.agent_pos = table1
        self.agent._update_knowledge()
        self.agent.step(4)  # Pick up obj1
        self.world.agent_pos = table3
        self.agent._update_knowledge()
        self.agent.step(4)  # Put down obj1
        
        # Move obj2 to table4  
        self.world.agent_pos = table2
        self.agent._update_knowledge()
        self.agent.step(4)  # Pick up obj2
        self.world.agent_pos = table4
        self.agent._update_knowledge()
        self.agent.step(4)  # Put down obj2
        
        # Verify both objects in correct locations
        kb = self.agent.get_knowledge()
        self.assertIn(obj1_id, kb['object_locations'])
        self.assertIn(obj2_id, kb['object_locations'])
        self.assertEqual(kb['object_locations'][obj1_id], self.world.room_ids[table3])
        self.assertEqual(kb['object_locations'][obj2_id], self.world.room_ids[table4])
        
        print("  ✓ Multiple objects test passed")
    
    def test_05_inventory_management_edge_cases(self):
        """Test inventory management edge cases and constraints"""
        print("\n=== Test 05: Inventory Management Edge Cases ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 3, "Need at least 3 tables for this test")
        
        table1, table2, table3 = table_positions[:3]
        obj1_id, obj2_id = ObjectType.KEY.value, ObjectType.TOOL.value
        
        # Setup
        self.world.grid[table1] = obj1_id
        self.world.grid[table2] = obj2_id
        self.world.grid[table3] = ObjectType.EMPTY.value
        
        # Test 1: Cannot pick up second object while holding first
        self.world.agent_pos = table1
        self.agent._update_knowledge()
        self.agent.step(4)  # Pick up obj1
        self.assertEqual(self.world.agent_inventory, obj1_id)
        
        self.world.agent_pos = table2
        self.agent._update_knowledge()
        self.agent.step(4)  # Try to pick up obj2 (should fail)
        self.assertEqual(self.world.agent_inventory, obj1_id)  # Still holding obj1
        
        print("  ✓ Cannot pick up second object while holding first")
        
        # Test 2: Cannot put down object on occupied table
        self.world.grid[table2] = obj2_id  # Ensure table2 has object
        self.world.agent_pos = table2
        self.agent._update_knowledge()
        self.agent.step(4)  # Try to put down (should fail - table occupied)
        self.assertEqual(self.world.agent_inventory, obj1_id)  # Still holding obj1
        self.assertEqual(self.world.grid[table2], obj2_id)  # Table still has obj2
        
        print("  ✓ Cannot put down object on occupied table")
        
        # Test 3: Can swap objects by putting down then picking up
        self.world.agent_pos = table3  # Empty table
        self.agent._update_knowledge()
        self.agent.step(4)  # Put down obj1
        self.assertEqual(self.world.agent_inventory, None)
        self.assertEqual(self.world.grid[table3], obj1_id)
        
        self.world.agent_pos = table2
        self.agent._update_knowledge()
        self.agent.step(4)  # Pick up obj2
        self.assertEqual(self.world.agent_inventory, obj2_id)
        self.assertEqual(self.world.grid[table2], ObjectType.EMPTY.value)
        
        print("  ✓ Object swapping works correctly")
    
    def test_06_room_exploration_and_mapping(self):
        """Test comprehensive room exploration and connectivity mapping"""
        print("\n=== Test 06: Room Exploration and Mapping ===")
        
        # Get all door transitions
        transitions = self.world.get_important_transitions()
        door_transitions = transitions['door_transitions']
        
        # Reset agent knowledge for clean exploration
        self.agent.knowledge_base['known_rooms'] = {self.world.get_current_room_id()}
        self.agent.knowledge_base['room_connections'] = set()
        
        # Explore using door transitions to visit all rooms
        visited_rooms = set()
        connections_discovered = set()
        
        for i, trans in enumerate(door_transitions[:12]):  # Limit to first 12 transitions
            if self.world.grid[trans['prev_position']] == 0:  # Valid position
                self.world.agent_pos = trans['prev_position']
                self.agent._update_knowledge()
                
                prev_room = self.agent.knowledge_base['current_room']
                self.agent.step(trans['action'])
                current_room = self.agent.knowledge_base['current_room']
                
                visited_rooms.add(current_room)
                
                if prev_room != current_room:
                    connection = tuple(sorted([prev_room, current_room]))
                    connections_discovered.add(connection)
        
        # Verify exploration results
        kb = self.agent.get_knowledge()
        
        # Should have visited multiple rooms
        self.assertGreater(len(visited_rooms), 1, "Should have visited multiple rooms")
        self.assertGreater(len(connections_discovered), 0, "Should have discovered connections")
        
        # Knowledge should match actual visits
        self.assertEqual(kb['known_rooms'], visited_rooms)
        
        # For connections, check that what we discovered is a subset of what agent knows
        # (agent might discover additional connections through different paths)
        for conn in connections_discovered:
            self.assertIn(conn, kb['room_connections'], 
                         f"Connection {conn} should be in agent's knowledge")
        
        # Test connectivity graph
        graph = self.agent.get_known_connectivity_graph()
        for room in visited_rooms:
            self.assertIn(room, graph)
        
        print(f"  ✓ Explored {len(visited_rooms)} rooms")
        print(f"  ✓ Discovered {len(connections_discovered)} connections")
        print(f"  ✓ Agent knows {len(kb['room_connections'])} total connections")
        print(f"  Rooms: {sorted(visited_rooms)}")
        print(f"  Connections discovered: {sorted(connections_discovered)}")
        print(f"  All connections known: {sorted(kb['room_connections'])}")
    
    def test_07_knowledge_persistence_consistency(self):
        """Test knowledge persistence and consistency through complex operations"""
        print("\n=== Test 07: Knowledge Persistence and Consistency ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 4, "Need at least 4 tables for this test")
        
        # Initial knowledge snapshot
        initial_kb = self.agent.get_knowledge()
        
        # Complex sequence of operations
        operations = [
            # Movement
            (1, "DOWN"), (3, "RIGHT"), (1, "DOWN"), (2, "LEFT"),
            # Object interactions
            (4, "TOGGLE"), (3, "RIGHT"), (4, "TOGGLE"),
            # More movement
            (0, "UP"), (2, "LEFT"), (0, "UP"), (3, "RIGHT"),
            # More interactions
            (4, "TOGGLE"), (1, "DOWN"), (4, "TOGGLE")
        ]
        
        # Manually place objects at strategic positions
        obj_id = ObjectType.TREASURE.value
        interaction_tables = [table_positions[0], table_positions[2]]
        self.world.grid[interaction_tables[0]] = obj_id
        self.world.grid[interaction_tables[1]] = ObjectType.EMPTY.value
        
        # Execute operations
        for i, (action, desc) in enumerate(operations):
            try:
                # Move to strategic tables for some operations
                if i == 4:  # First TOGGLE - should be at object table
                    self.world.agent_pos = interaction_tables[0]
                elif i == 6:  # Second TOGGLE - should be at empty table
                    self.world.agent_pos = interaction_tables[1]
                
                self.agent.step(action)
            except Exception as e:
                print(f"    Operation {i} ({desc}) failed: {e}")
        
        final_kb = self.agent.get_knowledge()
        
        # Knowledge consistency checks
        self.assertTrue(initial_kb['known_rooms'].issubset(final_kb['known_rooms']))
        self.assertTrue(initial_kb['room_connections'].issubset(final_kb['room_connections']))
        
        # Object location knowledge should be consistent with actual grid state
        for obj_value, room_id in final_kb['object_locations'].items():
            # Find the object in the grid and verify room matches
            found = False
            for table_pos in self.world.table_positions:
                if self.world.grid[table_pos] == obj_value:
                    actual_room = self.world.room_ids[table_pos]
                    self.assertEqual(room_id, actual_room, 
                                   f"Object {obj_value} knowledge says room {room_id} but actual room is {actual_room}")
                    found = True
                    break
            # Object might be in inventory, so not necessarily in grid
            if not found and self.world.agent_inventory != obj_value:
                self.fail(f"Object {obj_value} recorded in room {room_id} but not found in grid or inventory")
        
        print("  ✓ Knowledge remains consistent through complex operations")
        print(f"  ✓ Final knowledge: {len(final_kb['known_rooms'])} rooms, "
              f"{len(final_kb['room_connections'])} connections, "
              f"{len(final_kb['object_locations'])} object locations")
    
    def test_08_agent_query_methods_comprehensive(self):
        """Comprehensive test of all agent query methods"""
        print("\n=== Test 08: Agent Query Methods Comprehensive ===")
        
        # Build substantial knowledge base first
        table_positions = list(self.world.table_positions.keys())
        if len(table_positions) >= 4:
            # Place objects and interact
            obj1_id, obj2_id = ObjectType.KEY.value, ObjectType.FOOD.value
            self.world.grid[table_positions[0]] = obj1_id
            self.world.grid[table_positions[1]] = obj2_id
            self.world.grid[table_positions[2]] = ObjectType.EMPTY.value
            
            # Pick up and put down obj1
            self.world.agent_pos = table_positions[0]
            self.agent._update_knowledge()
            self.agent.step(4)  # Pick up
            self.world.agent_pos = table_positions[2]  
            self.agent._update_knowledge()
            self.agent.step(4)  # Put down
        
        # Explore some rooms
        transitions = self.world.get_important_transitions()
        for trans in transitions['door_transitions'][:6]:
            if self.world.grid[trans['prev_position']] == 0:
                self.world.agent_pos = trans['prev_position']
                self.agent._update_knowledge()
                self.agent.step(trans['action'])
        
        kb = self.agent.get_knowledge()
        
        # Test knows_room for all known rooms
        for room_id in kb['known_rooms']:
            self.assertTrue(self.agent.knows_room(room_id), 
                          f"Agent should know room {room_id}")
        
        # Test knows_connection for all recorded connections
        for room1, room2 in kb['room_connections']:
            self.assertTrue(self.agent.knows_connection(room1, room2),
                          f"Agent should know connection between {room1} and {room2}")
        
        # Test knows_object_location for all recorded objects
        for obj_id in kb['object_locations']:
            self.assertTrue(self.agent.knows_object_location(obj_id),
                          f"Agent should know location of object {obj_id}")
        
        # Test get_known_objects
        known_objects = self.agent.get_known_objects()
        self.assertEqual(set(known_objects), set(kb['object_locations'].keys()))
        
        # Test get_connected_rooms returns correct sets
        graph = self.agent.get_known_connectivity_graph()
        for room_id, connected_rooms in graph.items():
            manual_connections = set()
            for conn in kb['room_connections']:
                if room_id in conn:
                    other = conn[0] if conn[1] == room_id else conn[1]
                    manual_connections.add(other)
            self.assertEqual(connected_rooms, manual_connections,
                           f"Connectivity mismatch for room {room_id}")
        
        # Test get_known_connectivity_graph structure
        full_graph = self.agent.get_known_connectivity_graph()
        self.assertEqual(set(full_graph.keys()), kb['known_rooms'])
        
        print("  ✓ All query methods return correct information")
        print(f"  ✓ Known rooms: {sorted(kb['known_rooms'])}")
        print(f"  ✓ Known connections: {len(kb['room_connections'])}")
        print(f"  ✓ Known object locations: {len(kb['object_locations'])}")
    
    def test_09_performance_under_scale(self):
        """Test performance and correctness with larger scale operations"""
        print("\n=== Test 09: Performance Under Scale ===")
        
        # Test with many operations
        num_operations = 50
        successful_operations = 0
        
        for i in range(num_operations):
            action = random.randint(0, 4)  # Random action
            try:
                self.agent.step(action)
                successful_operations += 1
            except:
                pass  # Invalid moves are expected
        
        kb = self.agent.get_knowledge()
        
        # Should have executed many operations successfully
        self.assertGreater(successful_operations, num_operations * 0.5,
                          "Should successfully execute most operations")
        
        # Knowledge should remain consistent
        self.assertIsInstance(kb['known_rooms'], set)
        self.assertIsInstance(kb['room_connections'], set)
        self.assertIsInstance(kb['object_locations'], dict)
        
        # All known rooms should have valid room IDs
        for room_id in kb['known_rooms']:
            self.assertIsInstance(room_id, int)
            self.assertGreaterEqual(room_id, 0)
            self.assertLess(room_id, self.world.num_rooms)
        
        print(f"  ✓ Executed {successful_operations}/{num_operations} operations successfully")
        print(f"  ✓ Knowledge base remains consistent at scale")
        print(f"  ✓ Final state: {len(kb['known_rooms'])} rooms, "
              f"{len(kb['room_connections'])} connections")
        
    def test_10_integration_scenarios(self):
        """Test complex integration scenarios mimicking real usage"""
        print("\n=== Test 10: Integration Scenarios ===")
        
        table_positions = list(self.world.table_positions.keys())
        self.assertGreaterEqual(len(table_positions), 8, "Need at least 8 tables for this test")
        
        # Scenario: Treasure hunt - find and collect specific objects
        print("  Testing treasure hunt scenario...")
        
        # Reset agent to clean state
        self.world.agent_inventory = None
        self.agent.knowledge_base['previous_inventory'] = None
        
        # Place different objects around the world
        treasure_locations = {
            ObjectType.KEY: table_positions[0],
            ObjectType.TREASURE: table_positions[1], 
            ObjectType.FOOD: table_positions[2],
            ObjectType.TOOL: table_positions[3]
        }
        
        # Use separate stash locations for each object
        stash_locations = {
            ObjectType.KEY: table_positions[4],
            ObjectType.TREASURE: table_positions[5],
            ObjectType.FOOD: table_positions[6], 
            ObjectType.TOOL: table_positions[7]
        }
        
        # Place objects at treasure locations and clear all stash locations
        for obj_type, table_pos in treasure_locations.items():
            self.world.grid[table_pos] = obj_type.value
        
        for stash_pos in stash_locations.values():
            self.world.grid[stash_pos] = ObjectType.EMPTY.value
        
        collected_objects = []
        
        # "Collect" objects by moving to them and picking them up
        for obj_type, table_pos in treasure_locations.items():
            # Move to object
            self.world.agent_pos = table_pos
            self.agent._update_knowledge()
            
            # Verify we're at the right position and object exists
            self.assertEqual(self.world.agent_pos, table_pos)
            self.assertEqual(self.world.grid[table_pos], obj_type.value)
            self.assertIn(table_pos, self.world.table_positions)
            
            # Pick up object
            self.agent.step(4)
            
            # Verify pickup worked
            self.assertEqual(self.world.agent_inventory, obj_type.value, 
                            f"Failed to pick up object {obj_type.value} from {table_pos}")
            self.assertEqual(self.world.grid[table_pos], ObjectType.EMPTY.value)
            
            collected_objects.append(obj_type.value)
            
            # Move to appropriate stash for this object
            stash_pos = stash_locations[obj_type]
            self.world.agent_pos = stash_pos
            self.agent._update_knowledge()
            
            # Verify we're at stash and it's empty
            self.assertEqual(self.world.agent_pos, stash_pos)
            self.assertEqual(self.world.grid[stash_pos], ObjectType.EMPTY.value)
            
            # Put down in stash
            self.agent.step(4)
            
            # Verify putdown worked
            self.assertEqual(self.world.agent_inventory, None, 
                            f"Failed to put down object {obj_type.value} at {stash_pos}")
            self.assertEqual(self.world.grid[stash_pos], obj_type.value)
        
        # Verify all objects collected in their respective stashes
        for obj_type, stash_pos in stash_locations.items():
            self.assertEqual(self.world.grid[stash_pos], obj_type.value, 
                        f"Object {obj_type.value} should be in stash at {stash_pos}")
        
        # Verify knowledge reflects final state
        kb = self.agent.get_knowledge()
        # Should know location of all objects
        for obj_type in treasure_locations.keys():
            self.assertIn(obj_type.value, kb['object_locations'])
            self.assertEqual(kb['object_locations'][obj_type.value], 
                            self.world.room_ids[stash_locations[obj_type]])
        
        print("  ✓ Treasure hunt scenario completed successfully")
        print(f"  ✓ Collected {len(collected_objects)} objects")
        
        # Scenario: Exploration and mapping
        print("  Testing exploration and mapping scenario...")
        
        # Reset agent knowledge for clean test (but keep object locations)
        original_object_locations = kb['object_locations'].copy()
        self.agent.knowledge_base['known_rooms'] = {0}
        self.agent.knowledge_base['room_connections'] = set()
        self.agent.knowledge_base['current_room'] = 0
        self.agent.knowledge_base['object_locations'] = original_object_locations
        
        # Reset agent position and inventory
        self.world.agent_pos = (1, 1)
        self.world.agent_inventory = None
        self.agent._update_knowledge()
        
        # Use door transitions for guaranteed room exploration
        transitions = self.world.get_important_transitions()
        door_transitions = transitions['door_transitions']
        
        # Use door transitions to explore multiple rooms
        rooms_explored = set()
        connections_discovered = set()
        
        # Explore through several door transitions
        for i, trans in enumerate(door_transitions[:8]):  # Use first 8 door transitions
            if self.world.grid[trans['prev_position']] == 0:  # Valid position
                self.world.agent_pos = trans['prev_position']
                self.agent._update_knowledge()
                
                prev_room = self.agent.knowledge_base['current_room']
                self.agent.step(trans['action'])
                current_room = self.agent.knowledge_base['current_room']
                
                rooms_explored.add(current_room)
                
                if prev_room != current_room:
                    connection = tuple(sorted([prev_room, current_room]))
                    connections_discovered.add(connection)
        
        final_kb = self.agent.get_knowledge()
        
        # Should have explored multiple rooms (at least 2)
        self.assertGreater(len(final_kb['known_rooms']), 1, 
                        "Should have explored multiple rooms")
        self.assertGreater(len(final_kb['room_connections']), 0,
                        "Should have discovered at least one connection")
        
        print("  ✓ Exploration and mapping scenario completed successfully")
        print(f"  ✓ Mapped {len(final_kb['known_rooms'])} rooms with {len(final_kb['room_connections'])} connections")
def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=" * 70)
    print("COMPREHENSIVE GRIDWORLD AGENT TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGridWorldAgentComprehensive)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
    else:
        print("❌ Some tests failed")
        
        # Print failure details
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed test: {test}")
            print(f"Error: {traceback.splitlines()[-1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)