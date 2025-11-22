import numpy as np
import random
import unittest
from enum import Enum
from collections import deque, defaultdict
import heapq
from typing import Dict, List, Set, Tuple, Optional, Any
import copy

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


class ActionType(Enum):
    MOVE = 0
    PICK_UP = 1
    PUT_DOWN = 2


class PlanningDomain:
    """Fixed planning domain with corrected action logic"""
    
    def __init__(self):
        self.actions = {
            ActionType.MOVE: self._move_action,
            ActionType.PICK_UP: self._pick_up_action,
            ActionType.PUT_DOWN: self._put_down_action
        }
    
    def get_actions(self):
        """Get all available action types"""
        return list(self.actions.keys())
    
    def _move_action(self, state, from_room, to_room):
        """Move action: agent moves between connected rooms"""
        if state['agent_location'] != from_room:
            return None, f"Agent not in room {from_room} (currently in {state['agent_location']})"
            
        if to_room not in state['room_connections'].get(from_room, set()):
            return None, f"Rooms {from_room} and {to_room} are not connected"
            
        new_state = copy.deepcopy(state)
        new_state['agent_location'] = to_room
        return new_state, f"Moved from room {from_room} to room {to_room}"
    
    def _pick_up_action(self, state, object_id, room):
        """Pick up action: agent picks up object from table in current room"""
        if state['agent_location'] != room:
            return None, f"Agent not in room {room} (currently in {state['agent_location']})"
            
        if state['agent_inventory'] is not None:
            return None, f"Agent already holding object {state['agent_inventory']}"
            
        # FIX: Check if object is actually in this room
        if object_id not in state['object_locations']:
            return None, f"Object {object_id} location unknown"
            
        if state['object_locations'][object_id] != room:
            return None, f"Object {object_id} not in room {room} (it's in room {state['object_locations'][object_id]})"
            
        new_state = copy.deepcopy(state)
        new_state['agent_inventory'] = object_id
        # FIX: Remove the object from locations when picked up
        del new_state['object_locations'][object_id]
        return new_state, f"Picked up object {object_id} in room {room}"
    
    def _put_down_action(self, state, object_id, room):
        """Put down action: agent puts object on table in current room"""
        if state['agent_location'] != room:
            return None, f"Agent not in room {room}"
            
        if state['agent_inventory'] != object_id:
            return None, f"Agent not holding object {object_id} (holding {state['agent_inventory']})"
            
        if room not in state['tables']:
            return None, f"No table in room {room} to put object on"
            
        new_state = copy.deepcopy(state)
        new_state['agent_inventory'] = None
        # FIX: Add object to locations when put down
        new_state['object_locations'][object_id] = room
        return new_state, f"Put down object {object_id} in room {room}"
    
    def apply_action(self, state, action_type, **params):
        """Apply an action to a state and return new state"""
        if action_type not in self.actions:
            return None, f"Unknown action type: {action_type}"
            
        return self.actions[action_type](state, **params)
    
    def get_applicable_actions(self, state):
        """Get all applicable actions in current state"""
        applicable = []
        
        # Move actions
        current_room = state['agent_location']
        for connected_room in state['room_connections'].get(current_room, set()):
            applicable.append((ActionType.MOVE, {
                'from_room': current_room,
                'to_room': connected_room
            }))
        
        # Pick up actions
        if state['agent_inventory'] is None:
            for obj_id, obj_room in state['object_locations'].items():
                if obj_room == current_room:
                    applicable.append((ActionType.PICK_UP, {
                        'object_id': obj_id,
                        'room': current_room
                    }))
        
        # Put down action
        if state['agent_inventory'] is not None and current_room in state['tables']:
            applicable.append((ActionType.PUT_DOWN, {
                'object_id': state['agent_inventory'],
                'room': current_room
            }))
        
        return applicable
    
    def is_goal_state(self, state, goal):
        """Check if state satisfies goal condition"""
        if 'object_location' in goal:
            obj_id = goal['object_location']['object_id']
            target_room = goal['object_location']['room']
            return (obj_id in state['object_locations'] and 
                   state['object_locations'][obj_id] == target_room)
        return False


class Planner:
    """Fixed planner with better goal checking and plan validation"""
    
    def __init__(self, domain):
        self.domain = domain
    
    def bfs_plan(self, initial_state, goal, max_depth=50):
        """Find plan using BFS with proper goal checking"""
        # Check if goal is already satisfied in initial state
        if self.domain.is_goal_state(initial_state, goal):
            return []  # Return empty plan, not None
        
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue:
            state, plan = queue.popleft()
            
            # Check if we reached goal
            if self.domain.is_goal_state(state, goal):
                return plan
            
            # Check depth limit
            if len(plan) >= max_depth:
                continue
                
            # State fingerprint for visited check
            state_key = self._get_state_key(state)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Try all applicable actions
            for action_type, params in self.domain.get_applicable_actions(state):
                new_state, result_msg = self.domain.apply_action(state, action_type, **params)
                
                if new_state is not None:
                    # Verify this is actually a new state
                    new_state_key = self._get_state_key(new_state)
                    if new_state_key not in visited:
                        action_desc = f"{action_type.name}: {result_msg}"
                        queue.append((new_state, plan + [(action_type, params, action_desc)]))
        
        return None  # No plan found
    
    def _get_state_key(self, state):
        """Create a hashable key for state"""
        return (
            state['agent_location'],
            state['agent_inventory'],
            tuple(sorted(state['object_locations'].items()))
        )


# Fix the KnowledgeBasedPlanner's movement execution
class KnowledgeBasedPlanner:
    """Fixed knowledge-based planner with better movement execution"""
    
    def __init__(self, domain, agent):
        self.domain = domain
        self.agent = agent
        self.planner = Planner(domain)
    
    def create_planning_state_from_knowledge(self):
        """Create planning state from agent's current knowledge"""
        kb = self.agent.get_knowledge()
        world = self.agent.grid_world
        
        # Build room connections from knowledge
        room_connections = {}
        for conn in kb['room_connections']:
            room1, room2 = conn
            if room1 not in room_connections:
                room_connections[room1] = set()
            if room2 not in room_connections:
                room_connections[room2] = set()
            room_connections[room1].add(room2)
            room_connections[room2].add(room1)
        
        # Add known rooms that might not have connections yet
        for room in kb['known_rooms']:
            if room not in room_connections:
                room_connections[room] = set()
        
        # Find tables in known rooms
        tables = set()
        for table_pos, room_coords in world.table_positions.items():
            room_id = world.room_ids[table_pos]
            if room_id in kb['known_rooms']:
                tables.add(room_id)
        
        planning_state = {
            'agent_location': kb['current_room'],
            'agent_inventory': world.agent_inventory,
            'object_locations': kb['object_locations'].copy(),
            'room_connections': room_connections,
            'tables': tables
        }
        
        return planning_state
    
    def plan_with_current_knowledge(self, goal):
        """Create plan using current knowledge"""
        planning_state = self.create_planning_state_from_knowledge()
        
        # Verify the planning state is valid
        if not self._validate_planning_state(planning_state):
            return None
            
        return self.planner.bfs_plan(planning_state, goal)
    
    def _validate_planning_state(self, state):
        """Validate that the planning state is consistent"""
        # Check agent location is in known rooms
        if state['agent_location'] not in state['room_connections']:
            return False
            
        # Check object locations are in known rooms
        for obj_id, room_id in state['object_locations'].items():
            if room_id not in state['room_connections']:
                return False
                
        # Check agent inventory is valid
        if state['agent_inventory'] is not None and state['agent_inventory'] not in [ObjectType.KEY.value, ObjectType.TREASURE.value, ObjectType.FOOD.value, ObjectType.TOOL.value]:
            return False
            
        return True
    
    def execute_plan(self, plan, max_steps=100):
        """Execute a plan in the real environment - FIXED VERSION"""
        if not plan:
            return True, "No plan needed"
        
        steps = 0
        for i, (action_type, params, description) in enumerate(plan):
            if steps >= max_steps:
                return False, f"Plan execution timeout after {steps} steps"
            
            print(f"Step {i+1}: {description}")
            
            # Convert planning action to environment action
            if action_type == ActionType.MOVE:
                # Execute movement to target room
                success = self._execute_move_to_room(params['to_room'])
                if not success:
                    return False, f"Failed to move to room {params['to_room']}"
            elif action_type == ActionType.PICK_UP:
                # First ensure we're at a table position in the correct room
                if not self._ensure_at_table_in_room(params['room']):
                    return False, f"Cannot pick up - not at table in room {params['room']}"
                self.agent.step(4)  # TOGGLE action
            elif action_type == ActionType.PUT_DOWN:
                # First ensure we're at a table position in the correct room
                if not self._ensure_at_table_in_room(params['room']):
                    return False, f"Cannot put down - not at table in room {params['room']}"
                self.agent.step(4)  # TOGGLE action
            
            steps += 1
            
        return True, f"Plan executed successfully in {steps} steps"
    
    def _execute_move_to_room(self, target_room):
        """Execute movement to target room - SIMPLIFIED AND FIXED"""
        current_room = self.agent.get_knowledge()['current_room']
        
        if current_room == target_room:
            return True
            
        print(f"  Moving from room {current_room} to room {target_room}")
        
        # For demonstration purposes, use a simplified approach
        # Find any table position in the target room and move there
        target_table_pos = None
        for table_pos, room_coords in self.agent.grid_world.table_positions.items():
            if self.agent.grid_world.room_ids[table_pos] == target_room:
                target_table_pos = table_pos
                break
        
        if not target_table_pos:
            return False
        
        # Move to the target table position
        path = self._find_path_to_position(target_table_pos)
        if not path:
            return False
        
        # Execute the path
        for action in path:
            self.agent.step(action)
        
        # Verify we reached the target room
        new_room = self.agent.get_knowledge()['current_room']
        if new_room != target_room:
            print(f"  Warning: Expected room {target_room}, but ended up in room {new_room}")
            return False
            
        return True
    
    def _ensure_at_table_in_room(self, room):
        """Ensure agent is at a table position in the specified room"""
        current_room = self.agent.get_knowledge()['current_room']
        current_pos = self.agent.grid_world.agent_pos
        
        if current_room != room:
            return False
            
        # Check if we're at a table position
        return current_pos in self.agent.grid_world.table_positions
    
    def _find_path_to_position(self, target_pos):
        """Find path to a specific position using BFS"""
        start_pos = self.agent.grid_world.agent_pos
        
        if start_pos == target_pos:
            return []
        
        queue = deque([(start_pos, [])])
        visited = set([start_pos])
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == target_pos:
                return path
            
            x, y = pos
            for action, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.agent.grid_world.grid_size and 
                    0 <= ny < self.agent.grid_world.grid_size and
                    (nx, ny) not in visited and
                    self.agent.grid_world.grid[nx, ny] != 1):  # Not a wall
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [action]))
        
        return None

class GoalConditionedQLearning:
    def __init__(self, grid_world, learning_rate=0.1, discount_factor=0.9):
        self.grid_world = grid_world
        self.alpha = learning_rate
        self.gamma = discount_factor
        
        # Get all important transitions
        transitions = grid_world.get_important_transitions()
        self.all_transitions = transitions['door_transitions'] + transitions['object_transitions']
        
        # Create Q-tables for each transition goal
        self.q_tables = {}
        for i, transition in enumerate(self.all_transitions):
            self.q_tables[i] = defaultdict(lambda: np.zeros(5))  # 5 actions
        
        # Track learning progress
        self.q_value_sums = [[] for _ in range(len(self.all_transitions))]
    
    def get_state_key(self, position):
        """Convert position to hashable state key"""
        return position
    
    def choose_action(self, state_key):
        """Always choose random action (pure exploration)"""
        return random.randint(0, 4)
    
    def check_transition_activation(self, prev_state, action, next_state):
        """Check if any important transition was activated"""
        for i, transition in enumerate(self.all_transitions):
            if (transition['prev_position'] == prev_state and 
                transition['action'] == action and
                transition['next_position'] == next_state):
                return i  # Return transition index
        return None
    
    def learn_from_experience(self, prev_state, action, next_state, activated_transition):
        """Update ALL goal policies with the same experience"""
        prev_key = self.get_state_key(prev_state)
        next_key = self.get_state_key(next_state)
        
        for goal_index in range(len(self.all_transitions)):
            # Determine reward and terminal status for this goal
            if activated_transition == goal_index:
                # This is the goal transition for this policy
                reward = 1
                terminal = True  # Terminal for this specific goal's MDP
            elif activated_transition is not None:
                # Some other transition was activated - terminal with 0 reward
                reward = 0
                terminal = True
            else:
                # No transition activated - continue learning
                reward = 0
                terminal = False
            
            # Get current Q-value
            current_q = self.q_tables[goal_index][prev_key][action]
            
            if terminal:
                # Terminal state for this goal's MDP - no future rewards
                target = reward
            else:
                # Continue learning - bootstrap from next state
                next_max = np.max(self.q_tables[goal_index][next_key])
                target = reward + self.gamma * next_max
            
            # Update Q-value
            new_q = current_q + self.alpha * (target - current_q)
            self.q_tables[goal_index][prev_key][action] = new_q
    
    def get_q_value_sum(self, goal_index):
        """Calculate sum of all Q-values for a goal policy"""
        total = 0
        for state_actions in self.q_tables[goal_index].values():
            total += np.sum(state_actions)
        return total
    
    def get_random_valid_position(self):
        """Get a random position that is not a wall"""
        while True:
            x = random.randint(0, self.grid_world.grid_size - 1)
            y = random.randint(0, self.grid_world.grid_size - 1)
            if self.grid_world.grid[x, y] != 1:  # Not a wall
                return (x, y)
    
    def train_continuous(self, total_steps=1000000, log_interval=10000):
        """Continuous training without episodes - with longer duration"""
        print(f"Training {len(self.all_transitions)} goal-conditioned policies for {total_steps} steps...")
        
        # Start at a random valid position
        self.grid_world.agent_pos = self.get_random_valid_position()
        self.grid_world.agent_inventory = None
        
        prev_state = self.grid_world.agent_pos
        
        for step in range(total_steps):
            # Choose random action
            action = self.choose_action(prev_state)
            
            # Take action in the real environment
            self.grid_world.step(action)
            next_state = self.grid_world.agent_pos
            
            # Check if any transition was activated
            activated_transition = self.check_transition_activation(prev_state, action, next_state)
            
            # Learn from this experience for ALL goals
            self.learn_from_experience(prev_state, action, next_state, activated_transition)
            
            # Move to next state for continued exploration
            prev_state = next_state
            
            # Track learning progress
            if step % log_interval == 0:
                self._log_progress(step, total_steps)
    
    def _log_progress(self, step, total_steps):
        """Log learning progress using Q-value sums"""
        print(f"\nStep {step}/{total_steps}:")
        print("Goal Policy Progress (Sum of Q-values):")
        
        for goal_index in range(len(self.all_transitions)):
            q_sum = self.get_q_value_sum(goal_index)
            self.q_value_sums[goal_index].append(q_sum)
            
            goal_desc = self.get_goal_description(goal_index)
            print(f"  Goal {goal_index}: {q_sum:.2f} - {goal_desc}")
        
        # Show learning trends
        if step > 0:
            print("\nLearning Trends (change in Q-sum):")
            for goal_index in range(len(self.all_transitions)):
                if len(self.q_value_sums[goal_index]) >= 2:
                    current = self.q_value_sums[goal_index][-1]
                    previous = self.q_value_sums[goal_index][-2]
                    change = current - previous
                    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(f"  Goal {goal_index}: {change:+.2f} {trend}")
    
    def get_policy(self, goal_index, state):
        """Get best action for a given goal and state"""
        state_key = self.get_state_key(state)
        return np.argmax(self.q_tables[goal_index][state_key])
    
    def get_goal_description(self, goal_index):
        """Get description of a goal transition"""
        transition = self.all_transitions[goal_index]
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "TOGGLE"}
        action_name = action_names.get(transition['action'], str(transition['action']))
        return f"{transition['type']}: {transition['prev_position']} --{action_name}--> {transition['next_position']}"
    
    def evaluate_all_policies(self, num_tests=100, max_steps_per_test=500):
        """Evaluate each policy from multiple random starting positions"""
        print(f"\nEvaluating all policies with {num_tests} random starts each...")
        
        # Save current state to restore later
        original_pos = self.grid_world.agent_pos
        original_inventory = self.grid_world.agent_inventory
        
        success_rates = []
        avg_steps_to_success = []
        
        # Test each policy
        for goal_index in range(len(self.all_transitions)):
            print(f"\nTesting policy for goal {goal_index}: {self.get_goal_description(goal_index)}")
            
            successes = 0
            total_steps = 0
            
            for test_idx in range(num_tests):
                # Start from a random valid position
                self.grid_world.agent_pos = self.get_random_valid_position()
                self.grid_world.agent_inventory = None
                
                state = self.grid_world.agent_pos
                found_goal = False
                
                for step in range(max_steps_per_test):
                    # Use learned policy (greedy)
                    action = self.get_policy(goal_index, state)
                    
                    prev_state = state
                    self.grid_world.step(action)
                    state = self.grid_world.agent_pos
                    
                    # Check if goal transition was activated
                    activated = self.check_transition_activation(prev_state, action, state)
                    if activated == goal_index:
                        successes += 1
                        total_steps += step + 1
                        found_goal = True
                        break
                    elif activated is not None:
                        # Wrong transition activated
                        break
                
                if test_idx % 20 == 0:
                    print(f"  Test {test_idx+1}/{num_tests}: {successes} successes so far")
            
            success_rate = successes / num_tests
            avg_steps = total_steps / successes if successes > 0 else float('inf')
            
            success_rates.append(success_rate)
            avg_steps_to_success.append(avg_steps)
            
            print(f"  Final: {success_rate:.2f} success rate ({successes}/{num_tests})")
            if successes > 0:
                print(f"  Average steps to success: {avg_steps:.2f}")
        
        # Restore original state
        self.grid_world.agent_pos = original_pos
        self.grid_world.agent_inventory = original_inventory
        
        # Print summary
        print(f"\n=== EVALUATION SUMMARY ===")
        for goal_index in range(len(self.all_transitions)):
            print(f"Goal {goal_index}: {success_rates[goal_index]:.2f} success rate")
            if success_rates[goal_index] > 0:
                print(f"  Average steps: {avg_steps_to_success[goal_index]:.2f}")
            print(f"  Description: {self.get_goal_description(goal_index)}")
        
        return success_rates, avg_steps_to_success


class ActionOperators:
    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.operators = self._create_operators()
    
    def _create_operators(self):
        """Create list of tuples mapping high-level actions to low-level transitions"""
        operators = []
        
        # Get all transitions from the grid world
        transitions = self.grid_world.get_important_transitions()
        door_transitions = transitions['door_transitions']
        object_transitions = transitions['object_transitions']
        
        # MOVE action operators - through doors
        for transition in door_transitions:
            # Extract room information
            from_room = self.grid_world.room_ids[transition['prev_position']]
            to_room = self.grid_world.room_ids[transition['next_position']]
            
            # Preconditions for MOVE action
            preconditions = [
                f"agent_in_room({from_room})",
                f"connected({from_room}, {to_room})",
                f"at_position({transition['prev_position']})",
                f"facing_door({transition['prev_position']}, {transition['next_position']})"
            ]
            
            operators.append((
                "MOVE",
                preconditions,
                transition
            ))
        
        # PICK_UP action operators - at table positions
        for transition in object_transitions:
            room_id = self.grid_world.room_ids[transition['prev_position']]
            
            # Preconditions for PICK_UP action
            preconditions = [
                f"agent_in_room({room_id})",
                f"at_table({transition['prev_position']})",
                f"object_present({transition['prev_position']})",
                f"inventory_empty()"
            ]
            
            operators.append((
                "PICK_UP",
                preconditions,
                transition
            ))
        
        # PUT_DOWN action operators - at table positions
        for transition in object_transitions:
            room_id = self.grid_world.room_ids[transition['prev_position']]
            
            # Preconditions for PUT_DOWN action
            preconditions = [
                f"agent_in_room({room_id})",
                f"at_table({transition['prev_position']})", 
                f"table_empty({transition['prev_position']})",
                f"inventory_has_object()"
            ]
            
            operators.append((
                "PUT_DOWN",
                preconditions,
                transition
            ))
        
        return operators
    
    def get_operators_by_action(self, action_name):
        """Get all operators for a specific high-level action"""
        return [op for op in self.operators if op[0] == action_name]
    
    def get_operator_by_transition(self, transition):
        """Find operator that corresponds to a specific low-level transition"""
        for op in self.operators:
            if op[2] == transition:
                return op
        return None
    
    def find_applicable_operators(self, state):
        """Find operators whose preconditions are satisfied in current state"""
        applicable = []
        
        for operator in self.operators:
            action_name, preconditions, transition = operator
            
            # Check if preconditions are satisfied
            if self._check_preconditions(state, preconditions):
                applicable.append(operator)
        
        return applicable
    
    def _check_preconditions(self, state, preconditions):
        """Check if all preconditions are satisfied in the given state"""
        # Extract state information
        agent_pos = state['agent_position']
        agent_room = self.grid_world.room_ids.get(agent_pos, -1)
        agent_inventory = state.get('inventory', None)
        grid = state.get('grid', self.grid_world.grid)
        
        for precondition in preconditions:
            if precondition.startswith("agent_in_room("):
                # Extract room number from precondition
                room_num = int(precondition.split('(')[1].split(')')[0])
                if agent_room != room_num:
                    return False
                    
            elif precondition.startswith("at_position("):
                # Extract position from precondition
                pos_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                if agent_pos != (x, y):
                    return False
                    
            elif precondition.startswith("at_table("):
                # Check if at table position
                pos_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                if agent_pos != (x, y) or (x, y) not in self.grid_world.table_positions:
                    return False
                    
            elif precondition.startswith("object_present("):
                # Check if object is present at position
                pos_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                if grid[x, y] == 0:  # EMPTY
                    return False
                    
            elif precondition.startswith("table_empty("):
                # Check if table is empty
                pos_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                if grid[x, y] != 0:  # Not EMPTY
                    return False
                    
            elif precondition.startswith("inventory_empty()"):
                if agent_inventory is not None:
                    return False
                    
            elif precondition.startswith("inventory_has_object()"):
                if agent_inventory is None:
                    return False
                    
            elif precondition.startswith("connected("):
                # Check room connection (simplified - assumes all known connections are valid)
                rooms_str = precondition.split('(')[1].split(')')[0]
                room1, room2 = map(int, rooms_str.split(', '))
                # This would need access to the agent's knowledge base for proper checking
                pass  # For now, we'll assume this precondition is satisfied if we have the operator
                
            elif precondition.startswith("facing_door("):
                # Check if agent is positioned to move through door
                pass  # This would require checking the specific door orientation
        
        return True
    
    def display_operators(self):
        """Display all operators in a readable format"""
        print("=== High-Level Action Operators ===")
        
        # Group by action type
        actions = {}
        for op in self.operators:
            action_name = op[0]
            if action_name not in actions:
                actions[action_name] = []
            actions[action_name].append(op)
        
        for action_name, operators in actions.items():
            print(f"\n{action_name} Operators ({len(operators)}):")
            for i, op in enumerate(operators):
                action, preconditions, transition = op
                print(f"  {i}. Preconditions: {preconditions}")
                print(f"     Low-level: {transition['type']} at {transition['prev_position']} -> {transition['next_position']} via action {transition['action']}")
    
    def get_state_description(self):
        """Get current state for precondition checking"""
        return {
            'agent_position': self.grid_world.agent_pos,
            'inventory': self.grid_world.agent_inventory,
            'grid': self.grid_world.grid.copy()
        }

class IntegratedPlanner:
    """Planner that integrates high-level operators with learned low-level policies"""
    
    def __init__(self, grid_world, action_operators, q_learning_agent):
        self.grid_world = grid_world
        self.action_ops = action_operators
        self.q_agent = q_learning_agent
        self.domain = PlanningDomain()
        
        # Map low-level transitions to their operator indices
        self.transition_to_operator = {}
        for i, op in enumerate(self.action_ops.operators):
            transition_key = self._get_transition_key(op[2])
            self.transition_to_operator[transition_key] = i
    
    def _get_transition_key(self, transition):
        """Create a unique key for a transition"""
        return (transition['prev_position'], transition['action'], transition['next_position'])
    
    def _get_transition_index(self, transition):
        """Find the index of a transition in the Q-learning agent's list"""
        for i, t in enumerate(self.q_agent.all_transitions):
            if (t['prev_position'] == transition['prev_position'] and 
                t['action'] == transition['action'] and
                t['next_position'] == transition['next_position']):
                return i
        return None
    
    def is_operator_achievable(self, operator, current_state):
        """Check if an operator's low-level transition is achievable from current state"""
        _, _, transition = operator
        transition_index = self._get_transition_index(transition)
        
        if transition_index is None:
            return False
        
        # Check if max Q-value > 0 for current state
        current_pos = current_state['agent_position']
        max_q = np.max(self.q_agent.q_tables[transition_index][current_pos])
        
        return max_q > 0
    
    def get_achievable_operators(self, current_state):
        """Get all operators that are currently achievable"""
        applicable = self.action_ops.find_applicable_operators(current_state)
        achievable = []
        
        for op in applicable:
            if self.is_operator_achievable(op, current_state):
                achievable.append(op)
        
        return achievable
    
    def create_planning_state(self, current_state, activated_transition=None):
        """Create a planning state from current environment state"""
        if activated_transition is not None:
            # We've activated a transition - update position to the transition's end
            new_position = activated_transition['next_position']
        else:
            new_position = current_state['agent_position']
        
        room_id = self.grid_world.room_ids[new_position]
        
        # Build room connections from known transitions
        room_connections = defaultdict(set)
        for transition in self.q_agent.all_transitions:
            if transition['type'] == 'door':
                from_room = self.grid_world.room_ids[transition['prev_position']]
                to_room = self.grid_world.room_ids[transition['next_position']]
                room_connections[from_room].add(to_room)
                room_connections[to_room].add(from_room)
        
        # Find tables in rooms
        tables = set()
        for table_pos in self.grid_world.table_positions.keys():
            room_id = self.grid_world.room_ids[table_pos]
            tables.add(room_id)
        
        # Build object locations from current grid state
        object_locations = {}
        for table_pos in self.grid_world.table_positions.keys():
            x, y = table_pos
            if self.grid_world.grid[x, y] != 0:  # Not empty
                obj_id = self.grid_world.grid[x, y]
                room_id = self.grid_world.room_ids[table_pos]
                object_locations[obj_id] = room_id
        
        planning_state = {
            'agent_location': room_id,
            'agent_inventory': self.grid_world.agent_inventory,
            'object_locations': object_locations,
            'room_connections': dict(room_connections),
            'tables': tables,
            'low_level_position': new_position  # Track low-level position for execution
        }
        
        return planning_state
    
    def plan_with_learned_policies(self, goal, max_depth=20):
        """Create a plan using high-level operators that are achievable with learned policies"""
        current_state = self.action_ops.get_state_description()
        planning_state = self.create_planning_state(current_state)
        
        print("Starting planning with current state:")
        print(f"  Room: {planning_state['agent_location']}")
        print(f"  Position: {planning_state['low_level_position']}")
        print(f"  Inventory: {planning_state['agent_inventory']}")
        print(f"  Object locations: {planning_state['object_locations']}")
        
        # Use BFS to find plan
        queue = deque([(planning_state, [], current_state['agent_position'])])
        visited = set()
        
        while queue:
            state, plan, low_level_pos = queue.popleft()
            
            # Check if goal is satisfied
            if self.domain.is_goal_state(state, goal):
                return plan
            
            # Check depth limit
            if len(plan) >= max_depth:
                continue
            
            # State fingerprint for visited check
            state_key = (
                state['agent_location'],
                state['agent_inventory'],
                tuple(sorted(state['object_locations'].items()))
            )
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Get achievable operators in current low-level state
            current_low_level_state = {
                'agent_position': low_level_pos,
                'inventory': state['agent_inventory'],
                'grid': self.grid_world.grid.copy()
            }
            
            achievable_ops = self.get_achievable_operators(current_low_level_state)
            
            for op in achievable_ops:
                action_name, preconditions, transition = op
                transition_index = self._get_transition_index(transition)
                
                # Apply the operator to get new state
                if action_name == "MOVE":
                    # MOVE action changes room
                    from_room = self.grid_world.room_ids[transition['prev_position']]
                    to_room = self.grid_world.room_ids[transition['next_position']]
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_location'] = to_room
                    new_low_level_pos = transition['next_position']
                    
                    action_desc = f"MOVE from room {from_room} to room {to_room}"
                    params = {'from_room': from_room, 'to_room': to_room}
                    
                elif action_name == "PICK_UP":
                    # PICK_UP action - find object at current position
                    room_id = state['agent_location']
                    obj_id = None
                    for obj, obj_room in state['object_locations'].items():
                        if obj_room == room_id:
                            obj_id = obj
                            break
                    
                    if obj_id is None:
                        continue
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_inventory'] = obj_id
                    del new_state['object_locations'][obj_id]
                    new_low_level_pos = transition['next_position']  # Same position for toggle
                    
                    action_desc = f"PICK_UP object {obj_id} in room {room_id}"
                    params = {'object_id': obj_id, 'room': room_id}
                    
                elif action_name == "PUT_DOWN":
                    # PUT_DOWN action
                    if state['agent_inventory'] is None:
                        continue
                    
                    room_id = state['agent_location']
                    obj_id = state['agent_inventory']
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_inventory'] = None
                    new_state['object_locations'][obj_id] = room_id
                    new_low_level_pos = transition['next_position']  # Same position for toggle
                    
                    action_desc = f"PUT_DOWN object {obj_id} in room {room_id}"
                    params = {'object_id': obj_id, 'room': room_id}
                
                else:
                    continue
                
                # Add to plan
                new_plan = plan + [(action_name, params, action_desc, transition_index)]
                queue.append((new_state, new_plan, new_low_level_pos))
        
        return None  # No plan found
    
    def execute_integrated_plan(self, plan, max_steps_per_action=100):
        """Execute a plan using the learned low-level policies"""
        if not plan:
            return True, "No plan needed"
        
        print(f"\nExecuting plan with {len(plan)} steps:")
        
        for i, (action_name, params, description, transition_index) in enumerate(plan):
            print(f"\nStep {i+1}: {description}")
            
            # Use the learned policy to achieve the transition
            success = self._execute_transition_policy(transition_index, max_steps_per_action)
            
            if not success:
                return False, f"Failed to execute {description}"
            
            # Verify we achieved the expected state change
            if not self._verify_operator_applied(action_name, params):
                return False, f"Operator {action_name} did not produce expected result"
        
        return True, "Plan executed successfully"
    
    def _execute_transition_policy(self, transition_index, max_steps):
        """Execute the learned policy for a specific transition"""
        transition = self.q_agent.all_transitions[transition_index]
        target_position = transition['prev_position']
        
        # Move to the target position using the learned policy
        steps = 0
        while self.grid_world.agent_pos != target_position and steps < max_steps:
            # Get best action from current position for this transition
            current_pos = self.grid_world.agent_pos
            action = self.q_agent.get_policy(transition_index, current_pos)
            
            # Take the action
            self.grid_world.step(action)
            steps += 1
        
        if self.grid_world.agent_pos != target_position:
            return False
        
        # Now we're at the right position, take the transition action
        required_action = transition['action']
        self.grid_world.step(required_action)
        
        # Verify we reached the expected next position
        return self.grid_world.agent_pos == transition['next_position']
    
    def _verify_operator_applied(self, action_name, params):
        """Verify that the operator produced the expected state change"""
        current_state = self.action_ops.get_state_description()
        
        if action_name == "MOVE":
            expected_room = params['to_room']
            current_room = self.grid_world.room_ids[current_state['agent_position']]
            return current_room == expected_room
            
        elif action_name == "PICK_UP":
            expected_obj = params['object_id']
            return self.grid_world.agent_inventory == expected_obj
            
        elif action_name == "PUT_DOWN":
            expected_obj = params['object_id']
            expected_room = params['room']
            
            # Check if object is in the expected room
            for table_pos, room_coords in self.grid_world.table_positions.items():
                if self.grid_world.room_ids[table_pos] == expected_room:
                    if self.grid_world.grid[table_pos] == expected_obj:
                        return True
            return False
        
        return True

# Example usage and demonstration
def demonstrate_integrated_planning():
    # Create all components
    grid_world = GridWorld(num_rooms=25, room_size=5)
    
    # Train the low-level policies first
    print("=== Training Low-Level Policies ===")
    q_agent = GoalConditionedQLearning(grid_world, learning_rate=0.1, discount_factor=0.9)
    q_agent.train_continuous(total_steps=200000, log_interval=50000)
    
    # Create action operators
    print("\n=== Creating Action Operators ===")
    action_ops = ActionOperators(grid_world)
    action_ops.display_operators()
    
    # Create integrated planner
    print("\n=== Creating Integrated Planner ===")
    planner = IntegratedPlanner(grid_world, action_ops, q_agent)
    
    # Define a goal (e.g., put KEY in room 10)
    key_id = ObjectType.KEY.value
    goal = {
        'object_location': {
            'object_id': key_id,
            'room': 10
        }
    }
    
    print(f"\n=== Planning to achieve goal: KEY in room 10 ===")
    
    # Create a plan using integrated planning
    plan = planner.plan_with_learned_policies(goal)
    
    if plan:
        print(f"Found plan with {len(plan)} steps:")
        for i, (action_name, params, description, _) in enumerate(plan):
            print(f"  {i+1}. {description}")
        
        # Execute the plan
        print("\n=== Executing Plan ===")
        success, message = planner.execute_integrated_plan(plan)
        print(f"Result: {message}")
        
        # Verify goal achievement
        final_state = action_ops.get_state_description()
        final_planning_state = planner.create_planning_state(final_state)
        goal_achieved = planner.domain.is_goal_state(final_planning_state, goal)
        print(f"Goal achieved: {goal_achieved}")
        
    else:
        print("No plan found!")

def test_operator_achievability():
    """Test which operators are achievable from different positions"""
    grid_world = GridWorld(num_rooms=25, room_size=5)
    
    # Train policies briefly for testing
    q_agent = GoalConditionedQLearning(grid_world, learning_rate=0.1, discount_factor=0.9)
    q_agent.train_continuous(total_steps=50000, log_interval=25000)
    
    action_ops = ActionOperators(grid_world)
    planner = IntegratedPlanner(grid_world, action_ops, q_agent)
    
    # Test from different positions
    test_positions = [(1, 1), (5, 2), (10, 10)]
    
    for pos in test_positions:
        grid_world.agent_pos = pos
        current_state = action_ops.get_state_description()
        
        print(f"\nTesting from position {pos}:")
        achievable = planner.get_achievable_operators(current_state)
        
        print(f"  Achievable operators: {len(achievable)}")
        for op in achievable[:3]:  # Show first 3
            print(f"    - {op[0]} at {op[2]['prev_position']}")

if __name__ == "__main__":
    # Test operator achievability
    test_operator_achievability()
    
    # Run full integrated planning demonstration
    demonstrate_integrated_planning()