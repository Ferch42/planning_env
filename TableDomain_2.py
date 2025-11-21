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


class ExplorationPlanner:
    """Handles exploration when knowledge is incomplete"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def plan_exploration(self, target_object_id=None):
        """Plan exploration to discover unknown areas or find specific object"""
        kb = self.agent.get_knowledge()
        graph = self.agent.get_known_connectivity_graph()
        
        # Find unexplored connections or rooms with tables we haven't checked
        # For simplicity, return rooms to explore
        rooms_to_explore = []
        
        # Explore rooms that are connected but we haven't verified objects in
        for room in kb['known_rooms']:
            if target_object_id is None or target_object_id not in kb['object_locations']:
                rooms_to_explore.append(room)
        
        return rooms_to_explore[:3]  # Limit exploration scope


# Test the Planning Domain
class TestPlanningDomainSeparation(unittest.TestCase):
    """Test the properly separated planning domain"""
    
    def setUp(self):
        self.domain = PlanningDomain()
        self.world = GridWorld(num_rooms=9, room_size=5)
        self.agent = Agent(self.world)
    
    def test_01_pure_domain_actions(self):
        """Test pure planning domain actions"""
        print("\n=== Test 01: Pure Domain Actions ===")
        
        # Create a test state
        test_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {1: 0},  # Object 1 in room 0
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        # Test pick up action
        new_state, msg = self.domain.apply_action(
            test_state, ActionType.PICK_UP, object_id=1, room=0
        )
        self.assertIsNotNone(new_state)
        self.assertEqual(new_state['agent_inventory'], 1)
        self.assertNotIn(1, new_state['object_locations'])
        
        print("✓ Pure domain actions work correctly")
    
    def test_02_applicable_actions(self):
        """Test getting applicable actions"""
        print("\n=== Test 02: Applicable Actions ===")
        
        test_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {1: 0},
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        actions = self.domain.get_applicable_actions(test_state)
        self.assertGreater(len(actions), 0)
        
        # Should have move and pick up actions
        action_types = [action[0] for action in actions]
        self.assertIn(ActionType.MOVE, action_types)
        self.assertIn(ActionType.PICK_UP, action_types)
        
        print("✓ Applicable actions identified correctly")
    
    def test_03_planner_bfs(self):
        """Test BFS planner"""
        print("\n=== Test 03: BFS Planner ===")
        
        planner = Planner(self.domain)
        
        # Simple state where object is in same room
        test_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {1: 0},
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': 1, 'room': 1}}
        
        plan = planner.bfs_plan(test_state, goal)
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan), 0)
        
        print("✓ BFS planner found valid plan")
        for action_type, params, desc in plan:
            print(f"  - {desc}")
    
    def test_04_knowledge_based_planner(self):
        """Test knowledge-based planner"""
        print("\n=== Test 04: Knowledge-Based Planner ===")
        
        # Set up agent knowledge
        self.agent.knowledge_base['known_rooms'] = {0, 1}
        self.agent.knowledge_base['room_connections'] = {(0, 1)}
        self.agent.knowledge_base['object_locations'] = {1: 0}
        self.agent.knowledge_base['current_room'] = 0
        
        # Set up world state
        table_positions = list(self.world.table_positions.keys())
        if table_positions:
            # Place object in room 0
            for table_pos, room_coords in self.world.table_positions.items():
                if self.world.room_ids[table_pos] == 0:
                    self.world.grid[table_pos] = 1
                    break
        
        kb_planner = KnowledgeBasedPlanner(self.domain, self.agent)
        
        goal = {'object_location': {'object_id': 1, 'room': 1}}
        plan = kb_planner.plan_with_current_knowledge(goal)
        
        self.assertIsNotNone(plan)
        print("✓ Knowledge-based planner created plan")
    
    def test_05_exploration_planner(self):
        """Test exploration planner"""
        print("\n=== Test 05: Exploration Planner ===")
        
        # Set up partial knowledge
        self.agent.knowledge_base['known_rooms'] = {0, 1}
        self.agent.knowledge_base['room_connections'] = {(0, 1)}
        self.agent.knowledge_base['object_locations'] = {}  # No objects known
        
        exploration_planner = ExplorationPlanner(self.agent)
        exploration_plan = exploration_planner.plan_exploration(target_object_id=1)
        
        self.assertIsNotNone(exploration_plan)
        print("✓ Exploration planner created exploration plan")
    
    def test_06_goal_recognition(self):
        """Test goal state recognition"""
        print("\n=== Test 06: Goal Recognition ===")
        
        goal_state = {
            'agent_location': 1,
            'agent_inventory': None,
            'object_locations': {1: 1},  # Object 1 in room 1
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': 1, 'room': 1}}
        is_goal = self.domain.is_goal_state(goal_state, goal)
        self.assertTrue(is_goal)
        
        print("✓ Goal state correctly recognized")
    
    def test_07_invalid_actions(self):
        """Test handling of invalid actions"""
        print("\n=== Test 07: Invalid Actions ===")
        
        test_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {1: 1},  # Object in different room
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        # Try to pick up object from wrong room
        new_state, msg = self.domain.apply_action(
            test_state, ActionType.PICK_UP, object_id=1, room=0
        )
        self.assertIsNone(new_state)
        self.assertIn("not in room", msg)
        
        print("✓ Invalid actions properly rejected")


class RealisticPlanningTests(unittest.TestCase):
    """Fixed realistic tests that match actual planner behavior"""
    
    def setUp(self):
        self.world = GridWorld(num_rooms=9, room_size=5)
        self.agent = Agent(self.world)
        self.domain = PlanningDomain()
        self.planner = Planner(self.domain)
        self.kb_planner = KnowledgeBasedPlanner(self.domain, self.agent)
    
    def test_01_simple_delivery_known_location(self):
        """Test simple delivery when object location is known - FIXED"""
        print("\n=== Test 01: Simple Delivery Known Location (FIXED) ===")
        
        # Set up a simple known state where we need to move object from room 0 to room 1
        planning_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {ObjectType.KEY.value: 0},  # KEY in room 0
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 1}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        self.assertIsNotNone(plan, "Should find plan for simple delivery")
        
        # The correct plan should be: PICK_UP, MOVE, PUT_DOWN (3 actions)
        print(f"Plan found: {len(plan)} actions")
        for i, (action_type, params, desc) in enumerate(plan):
            print(f"  {i+1}. {desc}")
        
        # Verify plan execution leads to goal
        final_state = self._simulate_plan(planning_state, plan)
        self.assertTrue(self.domain.is_goal_state(final_state, goal))
        
        print("✓ Simple delivery planning works correctly")
    
    def test_02_goal_already_satisfied(self):
        """Test when goal is already satisfied - FIXED"""
        print("\n=== Test 02: Goal Already Satisfied (FIXED) ===")
        
        planning_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {ObjectType.KEY.value: 1},  # KEY already in target room
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 1}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        
        # When goal is already satisfied, should return empty plan, not None
        self.assertEqual(plan, [], "Should return empty plan when goal is already satisfied")
        
        print("✓ Correctly handles already satisfied goals")
    
    def test_03_agent_holding_object(self):
        """Test when agent is already holding the target object - FIXED"""
        print("\n=== Test 03: Agent Holding Object (FIXED) ===")
        
        planning_state = {
            'agent_location': 0,
            'agent_inventory': ObjectType.KEY.value,  # Agent already holding KEY
            'object_locations': {},  # No objects in world (agent has it)
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 1}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        self.assertIsNotNone(plan, "Should find plan when agent is holding object")
        
        print(f"Plan found: {len(plan)} actions")
        for i, (action_type, params, desc) in enumerate(plan):
            print(f"  {i+1}. {desc}")
        
        # Plan should be: move to room 1, put down (2 actions)
        final_state = self._simulate_plan(planning_state, plan)
        self.assertTrue(self.domain.is_goal_state(final_state, goal))
        
        print("✓ Planning works when agent is already holding object")
    
    def test_04_unreachable_object(self):
        """Test when object is in unreachable room - FIXED"""
        print("\n=== Test 04: Unreachable Object (FIXED) ===")
        
        planning_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {ObjectType.KEY.value: 2},  # Room 2 not connected
            'room_connections': {0: {1}, 1: {0}},  # No connection to room 2
            'tables': {0, 1, 2}
        }
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 0}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        self.assertIsNone(plan, "Should not find plan for unreachable object")
        
        print("✓ Correctly identifies unreachable objects")
    
    def test_05_knowledge_based_simple_delivery(self):
        """Test knowledge-based planning with simple scenario - FIXED"""
        print("\n=== Test 05: Knowledge-Based Simple Delivery (FIXED) ===")
        
        # Set up agent knowledge
        self.agent.knowledge_base.update({
            'known_rooms': {0, 1},
            'room_connections': {(0, 1)},
            'object_locations': {ObjectType.KEY.value: 0},
            'current_room': 0
        })
        
        # Set up actual world state to match knowledge
        for table_pos, room_coords in self.world.table_positions.items():
            room_id = self.world.room_ids[table_pos]
            if room_id == 0:
                self.world.grid[table_pos] = ObjectType.KEY.value
                # Place agent at the table in room 0
                self.world.agent_pos = table_pos
                break
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 1}}
        
        plan = self.kb_planner.plan_with_current_knowledge(goal)
        self.assertIsNotNone(plan, "Knowledge-based planner should find plan")
        
        print("Knowledge-based plan:")
        for i, (action_type, params, description) in enumerate(plan):
            print(f"  {i+1}. {description}")
        
        print("✓ Knowledge-based planning works for simple scenario")
    
    def test_06_action_validation(self):
        """Test that generated actions are actually valid - FIXED"""
        print("\n=== Test 06: Action Validation (FIXED) ===")
        
        # Create a state where object is in room 1, agent in room 0
        planning_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {ObjectType.KEY.value: 1},  # Object in room 1
            'room_connections': {0: {1}, 1: {0, 2}, 2: {1}},
            'tables': {0, 1, 2}
        }
        
        goal = {'object_location': {'object_id': ObjectType.KEY.value, 'room': 2}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        self.assertIsNotNone(plan, "Should find plan")
        
        # Verify every action is applicable when executed
        current_state = copy.deepcopy(planning_state)
        for i, (action_type, params, description) in enumerate(plan):
            print(f"  Step {i+1}: {description}")
            
            # Check if action is applicable
            applicable_actions = self.domain.get_applicable_actions(current_state)
            is_applicable = any(
                app_action_type == action_type and app_params == params
                for app_action_type, app_params in applicable_actions
            )
            
            self.assertTrue(is_applicable, 
                          f"Action {i+1} should be applicable: {description}\n"
                          f"Current state: agent_loc={current_state['agent_location']}, "
                          f"inventory={current_state['agent_inventory']}, "
                          f"obj_locs={current_state['object_locations']}")
            
            # Apply action
            new_state, msg = self.domain.apply_action(current_state, action_type, **params)
            self.assertIsNotNone(new_state, f"Action {i+1} should succeed: {msg}")
            current_state = new_state
        
        # Verify goal is achieved
        self.assertTrue(self.domain.is_goal_state(current_state, goal))
        
        print("✓ All generated actions are valid and executable")
    
    def test_07_multiple_objects_same_room(self):
        """Test planning with multiple objects in the same room - FIXED"""
        print("\n=== Test 07: Multiple Objects Same Room (FIXED) ===")
        
        # FIX: Both objects are in room 0
        planning_state = {
            'agent_location': 0,
            'agent_inventory': None,
            'object_locations': {
                ObjectType.KEY.value: 0,      # KEY in room 0
                ObjectType.TREASURE.value: 0  # TREASURE in room 0
            },
            'room_connections': {0: {1}, 1: {0}},
            'tables': {0, 1}
        }
        
        goal = {'object_location': {'object_id': ObjectType.TREASURE.value, 'room': 1}}
        
        plan = self.planner.bfs_plan(planning_state, goal)
        self.assertIsNotNone(plan, "Should find plan with multiple objects")
        
        print("Plan for multiple objects:")
        for i, (action_type, params, desc) in enumerate(plan):
            print(f"  {i+1}. {desc}")
        
        # Plan should pick up treasure, not key
        pick_up_actions = [action for action in plan if action[0] == ActionType.PICK_UP]
        if pick_up_actions:
            self.assertEqual(pick_up_actions[0][1]['object_id'], ObjectType.TREASURE.value)
        
        final_state = self._simulate_plan(planning_state, plan)
        self.assertTrue(self.domain.is_goal_state(final_state, goal))
        
        print("✓ Correctly handles multiple objects in same room")
    
    def _simulate_plan(self, initial_state, plan):
        """Simulate executing a plan and return the final state"""
        current_state = copy.deepcopy(initial_state)
        for action_type, params, description in plan:
            new_state, msg = self.domain.apply_action(current_state, action_type, **params)
            if new_state is None:
                self.fail(f"Action failed during simulation: {action_type} with {params}. Message: {msg}")
            current_state = new_state
        return current_state


# Update the demonstration function to use the fixed planner
def demonstrate_separated_planning():
    """Demonstrate the properly separated planning system"""
    print("\n" + "=" * 70)
    print("SEPARATED PLANNING SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Create components
    domain = PlanningDomain()
    world = GridWorld(num_rooms=9, room_size=5)
    agent = Agent(world)
    
    # Set up a known state for planning where we actually need to move the object
    planning_state = {
        'agent_location': 0,
        'agent_inventory': None,
        'object_locations': {
            ObjectType.KEY.value: 0,  # KEY in room 0
            ObjectType.TREASURE.value: 2  # TREASURE in room 2
        },
        'room_connections': {
            0: {1, 2},
            1: {0, 3},
            2: {0, 4},
            3: {1},
            4: {2}
        },
        'tables': {0, 1, 2, 3, 4}
    }
    
    # Define goal: deliver KEY to room 4 (NOT where it currently is)
    goal = {
        'object_location': {
            'object_id': ObjectType.KEY.value,
            'room': 4
        }
    }
    
    # Plan using pure domain
    planner = Planner(domain)
    plan = planner.bfs_plan(planning_state, goal)
    
    if plan:
        print("Pure planning domain plan:")
        for i, (action_type, params, description) in enumerate(plan):
            print(f"  {i+1}. {description}")
    else:
        print("No plan found with pure planning domain")
    
    # Demonstrate knowledge-based planning
    print("\nKnowledge-based planning:")
    kb_planner = KnowledgeBasedPlanner(domain, agent)
    
    # Set up agent knowledge to match our planning state
    print("Building agent knowledge through exploration...")
    _build_agent_knowledge(agent, world)
    
    # Now set up the specific knowledge for our scenario
    # KEY is in room 0, we want to deliver to room 4
    agent.knowledge_base.update({
        'known_rooms': set(planning_state['room_connections'].keys()),
        'object_locations': {ObjectType.KEY.value: 0},  # Only KEY in room 0
        'current_room': planning_state['agent_location']
    })
    
    # Convert connections to the format agent expects
    for room, connections in planning_state['room_connections'].items():
        for connected_room in connections:
            agent.knowledge_base['room_connections'].add(
                tuple(sorted([room, connected_room]))
            )
    
    # Also set up the actual world state to match
    _setup_world_state(world, planning_state)
    
    print("Agent knowledge after setup:")
    agent.render_knowledge()
    
    print(f"\nGoal: Deliver KEY (object {ObjectType.KEY.value}) to room 4")
    
    kb_plan = kb_planner.plan_with_current_knowledge(goal)
    
    if kb_plan:
        print("Knowledge-based plan:")
        for i, (action_type, params, description) in enumerate(kb_plan):
            print(f"  {i+1}. {description}")
        
        # Try to execute the plan (simulated)
        print("\nExecuting plan...")
        success, message = kb_planner.execute_plan(kb_plan)
        print(f"Execution result: {message}")
        
        if success:
            # Show final state
            print("\nFinal world state:")
            world.render()
            print("\nFinal agent knowledge:")
            agent.render_knowledge()
    else:
        print("No plan found with knowledge-based planning")
    
    # Demonstrate exploration planning
    print("\nExploration planning:")
    exploration_planner = ExplorationPlanner(agent)
    exploration_targets = exploration_planner.plan_exploration(
        target_object_id=ObjectType.TOOL.value
    )
    print(f"Exploration targets: {exploration_targets}")

def _build_agent_knowledge(agent, world):
    """Build some basic knowledge by exploring"""
    transitions = world.get_important_transitions()
    for trans in transitions['door_transitions'][:4]:
        if world.grid[trans['prev_position']] == 0:
            world.agent_pos = trans['prev_position']
            agent._update_knowledge()
            agent.step(trans['action'])

def _setup_world_state(world, planning_state):
    """Set up the actual world state to match the planning state"""
    # Set agent position to a table in the starting room
    for table_pos, room_coords in world.table_positions.items():
        if world.room_ids[table_pos] == planning_state['agent_location']:
            world.agent_pos = table_pos
            break
    
    # Clear all tables first
    for table_pos in world.table_positions:
        world.grid[table_pos] = ObjectType.EMPTY.value
    
    # Place objects according to planning state
    for obj_id, room_id in planning_state['object_locations'].items():
        for table_pos, room_coords in world.table_positions.items():
            if world.room_ids[table_pos] == room_id:
                world.grid[table_pos] = obj_id
                break


def run_separated_planning_tests():
    """Run tests for separated planning system"""
    print("=" * 70)
    print("SEPARATED PLANNING DOMAIN TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPlanningDomainSeparation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SEPARATED PLANNING TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 ALL SEPARATED PLANNING TESTS PASSED!")
    else:
        print("❌ Some separated planning tests failed")
        
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed test: {test}")
            print(f"Error: {traceback.splitlines()[-1]}")
    
    return result.wasSuccessful()


def run_realistic_planning_tests():
    """Run the realistic planning tests"""
    print("=" * 70)
    print("REALISTIC PLANNING TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(RealisticPlanningTests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("REALISTIC PLANNING TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 ALL REALISTIC PLANNING TESTS PASSED!")
        print("✅ The planning system is working correctly for realistic scenarios!")
    else:
        print("❌ Some realistic planning tests failed")
        
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed test: {test}")
            error_line = traceback.splitlines()[-1]
            print(f"Error: {error_line}")
    
    return result.wasSuccessful()


# Main execution
if __name__ == "__main__":
    # Run the original separated planning tests
    original_success = run_separated_planning_tests()
    
    # Run the realistic tests
    realistic_success = run_realistic_planning_tests()
    
    # Demonstrate the system
    demonstrate_separated_planning()
    
    # Exit with appropriate code
    exit(0 if (original_success and realistic_success) else 1)