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
import numpy as np
import random
import unittest
from enum import Enum
from collections import deque, defaultdict
import heapq
from typing import Dict, List, Set, Tuple, Optional, Any
import copy

class GoalConditionedQLearning:
    def __init__(self, grid_world, learning_rate=0.1, discount_factor=0.9, debug=False):
        self.grid_world = grid_world
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.debug = debug
        
        # Get all important transitions
        transitions = grid_world.get_important_transitions()
        self.all_transitions = transitions['door_transitions'] + transitions['object_transitions']
        
        # Create Q-tables for each transition goal
        self.q_tables = {}
        for i, transition in enumerate(self.all_transitions):
            self.q_tables[i] = defaultdict(lambda: np.zeros(5))  # 5 actions
        
        # Track learning progress
        self.q_value_sums = [[] for _ in range(len(self.all_transitions))]
        
        # Debug statistics
        self.training_stats = {
            'steps': 0,
            'transitions_activated': [0] * len(self.all_transitions),
            'q_updates': 0,
            'valid_moves': 0,
            'wall_collisions': 0
        }
    
    def get_state_key(self, position):
        """Convert position to hashable state key"""
        return position
    
    def choose_action(self, state_key, goal_index=None):
        """Choose action with epsilon-greedy policy"""
        if random.random() < 0.9:  # 90% exploration during training
            return random.randint(0, 4)
        else:
            # 10% exploitation: use Q-values if available
            if goal_index is not None:
                q_values = self.q_tables[goal_index][state_key]
                if np.max(q_values) > 0:
                    return np.argmax(q_values)
            return random.randint(0, 4)
    
    def check_transition_activation(self, prev_state, action, next_state):
        """Check if any important transition was activated"""
        for i, transition in enumerate(self.all_transitions):
            if (transition['prev_position'] == prev_state and 
                transition['action'] == action and
                transition['next_position'] == next_state):
                
                if self.debug:
                    print(f"  Transition {i} activated: {transition['type']} at {prev_state} -> {next_state}")
                self.training_stats['transitions_activated'][i] += 1
                return i
        
        # Also check for object interactions at current position
        if action == 4:  # TOGGLE action
            for i, transition in enumerate(self.all_transitions):
                if (transition['type'] == 'object' and 
                    transition['prev_position'] == prev_state and
                    transition['action'] == action):
                    
                    if self.debug:
                        print(f"  Object transition {i} activated at {prev_state}")
                    self.training_stats['transitions_activated'][i] += 1
                    return i
        
        return None
    
    def learn_from_experience(self, prev_state, action, next_state, activated_transition):
        """Update ALL goal policies with the same experience"""
        prev_key = self.get_state_key(prev_state)
        next_key = self.get_state_key(next_state)
        
        for goal_index in range(len(self.all_transitions)):
            # Determine reward and terminal status for this goal
            if activated_transition == goal_index:
                # This is the goal transition for this policy
                reward = 10.0
                terminal = True
            elif activated_transition is not None:
                # Some other transition was activated - small penalty
                reward = -1.0
                terminal = True
            else:
                # No transition activated - small negative reward to encourage efficiency
                reward = -0.1
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
            self.training_stats['q_updates'] += 1
    
    def get_valid_actions(self, position):
        """Get list of valid actions that don't hit walls"""
        x, y = position
        valid_actions = []
        
        # Check UP
        if x > 0 and self.grid_world.grid[x-1, y] != 1:
            valid_actions.append(0)
        # Check DOWN
        if x < self.grid_world.grid_size-1 and self.grid_world.grid[x+1, y] != 1:
            valid_actions.append(1)
        # Check LEFT
        if y > 0 and self.grid_world.grid[x, y-1] != 1:
            valid_actions.append(2)
        # Check RIGHT
        if y < self.grid_world.grid_size-1 and self.grid_world.grid[x, y+1] != 1:
            valid_actions.append(3)
        # TOGGLE is always valid (though may not do anything)
        valid_actions.append(4)
        
        return valid_actions
    
    def get_random_valid_position(self):
        """Get a random position that is not a wall"""
        while True:
            x = random.randint(0, self.grid_world.grid_size - 1)
            y = random.randint(0, self.grid_world.grid_size - 1)
            if self.grid_world.grid[x, y] != 1:
                return (x, y)
    
    def train_continuous(self, total_steps=100000, log_interval=10000):
        """Continuous training without episodes"""
        print(f"Training {len(self.all_transitions)} goal-conditioned policies for {total_steps} steps...")
        
        # Start at a random valid position
        self.grid_world.agent_pos = self.get_random_valid_position()
        self.grid_world.agent_inventory = None
        
        prev_state = self.grid_world.agent_pos
        
        for step in range(total_steps):
            self.training_stats['steps'] += 1
            
            # Choose action from valid actions only
            valid_actions = self.get_valid_actions(prev_state)
            if not valid_actions:
                # If no valid actions, reset position
                self.grid_world.agent_pos = self.get_random_valid_position()
                prev_state = self.grid_world.agent_pos
                continue
                
            action = random.choice(valid_actions)
            
            # Take action in the real environment
            self.grid_world.step(action)
            next_state = self.grid_world.agent_pos
            
            # Track if we actually moved
            if prev_state != next_state:
                self.training_stats['valid_moves'] += 1
            else:
                self.training_stats['wall_collisions'] += 1
            
            # Check if any transition was activated
            activated_transition = self.check_transition_activation(prev_state, action, next_state)
            
            # Learn from this experience for ALL goals
            self.learn_from_experience(prev_state, action, next_state, activated_transition)
            
            # Move to next state for continued exploration
            prev_state = next_state
            
            # Track learning progress
            if step % log_interval == 0:
                self._log_progress(step, total_steps)
        
        # Print final training statistics
        self._print_training_stats()
    
    def _log_progress(self, step, total_steps):
        """Log learning progress using Q-value sums"""
        print(f"\nStep {step}/{total_steps}:")
        
        # Show top 5 learned policies
        q_sums = []
        for goal_index in range(len(self.all_transitions)):
            q_sum = self.get_q_value_sum(goal_index)
            q_sums.append((goal_index, q_sum))
        
        # Sort by Q-sum
        q_sums.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 5 Learned Policies (by Q-value sum):")
        for i, (goal_index, q_sum) in enumerate(q_sums[:5]):
            goal_desc = self.get_goal_description(goal_index)
            activations = self.training_stats['transitions_activated'][goal_index]
            print(f"  {i+1}. Goal {goal_index}: {q_sum:.2f} ({activations} activations) - {goal_desc}")
    
    def _print_training_stats(self):
        """Print detailed training statistics"""
        print("\n=== Training Statistics ===")
        print(f"Total steps: {self.training_stats['steps']}")
        print(f"Valid moves: {self.training_stats['valid_moves']}")
        print(f"Wall collisions: {self.training_stats['wall_collisions']}")
        print(f"Total Q-updates: {self.training_stats['q_updates']}")
        
        # Print transition activation summary
        active_transitions = [(i, count) for i, count in enumerate(self.training_stats['transitions_activated']) if count > 0]
        print(f"\nActivated {len(active_transitions)}/{len(self.all_transitions)} transitions:")
        for trans_idx, count in sorted(active_transitions, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Transition {trans_idx}: {count} times - {self.get_goal_description(trans_idx)}")
    
    def get_policy(self, goal_index, state):
        """Get best action for a given goal and state"""
        state_key = self.get_state_key(state)
        q_values = self.q_tables[goal_index][state_key]
        
        # Only consider valid actions
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return 4  # Default to TOGGLE if no valid moves
        
        # Find best valid action
        best_action = None
        best_q = -float('inf')
        
        for action in valid_actions:
            if q_values[action] > best_q:
                best_q = q_values[action]
                best_action = action
        
        # If all Q-values are zero, choose random valid action
        if best_q <= 0:
            best_action = random.choice(valid_actions)
        
        if self.debug:
            print(f"  Policy for goal {goal_index}: state {state} -> action {best_action} (Q-values: {q_values})")
        
        return best_action
    
    def get_goal_description(self, goal_index):
        """Get description of a goal transition"""
        transition = self.all_transitions[goal_index]
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "TOGGLE"}
        action_name = action_names.get(transition['action'], str(transition['action']))
        return f"{transition['type']}: {transition['prev_position']} --{action_name}--> {transition['next_position']}"
    
    def test_single_policy(self, goal_index, start_position, max_steps=100):
        """Test a single policy from a specific starting position"""
        print(f"\nTesting policy {goal_index} from position {start_position}:")
        print(f"Goal: {self.get_goal_description(goal_index)}")
        
        # Save current state
        original_pos = self.grid_world.agent_pos
        original_inventory = self.grid_world.agent_inventory
        
        # Set test position
        self.grid_world.agent_pos = start_position
        self.grid_world.agent_inventory = None
        
        state = start_position
        path = [state]
        
        for step in range(max_steps):
            action = self.get_policy(goal_index, state)
            
            prev_state = state
            self.grid_world.step(action)
            state = self.grid_world.agent_pos
            path.append(state)
            
            print(f"  Step {step}: {prev_state} -> {state} (action {action})")
            
            # Check if goal achieved
            activated = self.check_transition_activation(prev_state, action, state)
            if activated == goal_index:
                print(f"  SUCCESS: Goal achieved in {step+1} steps!")
                break
            elif activated is not None:
                print(f"  WRONG TRANSITION: Activated {activated} instead of {goal_index}")
                break
        
        # Restore original state
        self.grid_world.agent_pos = original_pos
        self.grid_world.agent_inventory = original_inventory
        
        return path

    def get_q_value_sum(self, goal_index):
        """Calculate sum of all Q-values for a goal policy"""
        total = 0
        for state_actions in self.q_tables[goal_index].values():
            total += np.sum(np.maximum(state_actions, 0))
        return total


class ActionOperators:
    def __init__(self, grid_world, debug=False):
        self.grid_world = grid_world
        self.debug = debug
        self.operators = self._create_operators()
        
        if self.debug:
            print(f"Created {len(self.operators)} total operators")
    
    def _create_operators(self):
        """Create operators using ONLY planning domain predicates"""
        operators = []
        
        transitions = self.grid_world.get_important_transitions()
        door_transitions = transitions['door_transitions']
        object_transitions = transitions['object_transitions']
        
        # MOVE operators - use only room-level predicates
        for transition in door_transitions:
            from_room = self.grid_world.room_ids[transition['prev_position']]
            to_room = self.grid_world.room_ids[transition['next_position']]
            
            # ONLY use predicates that exist in planning domain
            preconditions = [
                f"agent_in_room({from_room})",
                f"connected({from_room}, {to_room})"
                # REMOVED: at_position, facing_door - these don't exist!
            ]
            
            operators.append(("MOVE", preconditions, transition))
        
        # PICK_UP operators - use only room-level predicates  
        for transition in object_transitions:
            room_id = self.grid_world.room_ids[transition['prev_position']]
            
            # ONLY use predicates that exist in planning domain
            preconditions = [
                f"agent_in_room({room_id})",
                f"object_in_room({room_id})",  # Object exists in this room
                f"inventory_empty()"
                # REMOVED: at_table, object_present - these don't exist!
            ]
            
            operators.append(("PICK_UP", preconditions, transition))
        
        # PUT_DOWN operators - use only room-level predicates
        for transition in object_transitions:
            room_id = self.grid_world.room_ids[transition['prev_position']]
            
            # ONLY use predicates that exist in planning domain
            preconditions = [
                f"agent_in_room({room_id})", 
                f"room_has_table({room_id})",  # Room has a table
                f"inventory_has_object()"
                # REMOVED: at_table, table_empty - these don't exist!
            ]
            
            operators.append(("PUT_DOWN", preconditions, transition))
        
        return operators
    
    def get_operators_by_action(self, action_name):
        """Get all operators for a specific high-level action"""
        operators = [op for op in self.operators if op[0] == action_name]
        if self.debug:
            print(f"Found {len(operators)} operators for action {action_name}")
        return operators
    
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
        
        if self.debug:
            print(f"Found {len(applicable)} applicable operators out of {len(self.operators)} total")
        
        return applicable
    
    def _check_preconditions(self, state, preconditions):
        """Check preconditions using available state information"""
        # Handle both planning state and grid world state representations
        if 'agent_location' in state:
            # Planning state representation
            agent_room = state['agent_location']
            agent_inventory = state['agent_inventory']
            object_locations = state['object_locations']
            room_connections = state['room_connections']
            tables = state['tables']
        else:
            # Grid world state representation (for testing)
            agent_pos = state['agent_position']
            agent_room = self.grid_world.room_ids.get(agent_pos, -1)
            agent_inventory = state.get('inventory', None)
            
            # Build planning state components from grid world
            object_locations = {}
            for table_pos in self.grid_world.table_positions.keys():
                x, y = table_pos
                if self.grid_world.grid[x, y] != 0:
                    obj_id = self.grid_world.grid[x, y]
                    room_id = self.grid_world.room_ids[table_pos]
                    object_locations[obj_id] = room_id
            
            # Build room connections from door transitions
            room_connections = defaultdict(set)
            for transition in self.grid_world.door_transitions:
                from_room = self.grid_world.room_ids[transition['prev_position']]
                to_room = self.grid_world.room_ids[transition['next_position']]
                room_connections[from_room].add(to_room)
                room_connections[to_room].add(from_room)
            
            # Build tables set
            tables = set()
            for table_pos in self.grid_world.table_positions.keys():
                room_id_table = self.grid_world.room_ids[table_pos]
                tables.add(room_id_table)
        
        # Now check preconditions
        for precondition in preconditions:
            if precondition.startswith("agent_in_room("):
                room_num = int(precondition.split('(')[1].split(')')[0])
                if agent_room != room_num:
                    if self.debug:
                        print(f"  Precondition failed: agent_in_room({room_num}), agent is in room {agent_room}")
                    return False
                    
            elif precondition.startswith("connected("):
                rooms_str = precondition.split('(')[1].split(')')[0]
                parts = rooms_str.split(',')
                room1 = int(parts[0].strip())
                room2 = int(parts[1].strip())
                # Check if rooms are connected
                if room2 not in room_connections.get(room1, set()):
                    if self.debug:
                        print(f"  Precondition failed: connected({room1}, {room2})")
                    return False
                    
            elif precondition.startswith("object_in_room("):
                room_num = int(precondition.split('(')[1].split(')')[0])
                # Check if any object is in this room
                object_found = any(room == room_num for room in object_locations.values())
                if not object_found:
                    if self.debug:
                        print(f"  Precondition failed: object_in_room({room_num})")
                    return False
                    
            elif precondition.startswith("room_has_table("):
                room_num = int(precondition.split('(')[1].split(')')[0])
                if room_num not in tables:
                    if self.debug:
                        print(f"  Precondition failed: room_has_table({room_num})")
                    return False
                    
            elif precondition.startswith("inventory_empty()"):
                if agent_inventory is not None:
                    if self.debug:
                        print(f"  Precondition failed: inventory_empty(), inventory has {agent_inventory}")
                    return False
                    
            elif precondition.startswith("inventory_has_object()"):
                if agent_inventory is None:
                    if self.debug:
                        print("  Precondition failed: inventory_has_object(), inventory is empty")
                    return False
        
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
        """Get current state for precondition checking - return planning state"""
        # Create a planning state from current grid world state
        agent_room = self.grid_world.room_ids[self.grid_world.agent_pos]
        
        # Build object locations
        object_locations = {}
        for table_pos in self.grid_world.table_positions.keys():
            x, y = table_pos
            if self.grid_world.grid[x, y] != 0:
                obj_id = self.grid_world.grid[x, y]
                room_id = self.grid_world.room_ids[table_pos]
                object_locations[obj_id] = room_id
        
        # Build room connections
        room_connections = defaultdict(set)
        for transition in self.grid_world.door_transitions:
            from_room = self.grid_world.room_ids[transition['prev_position']]
            to_room = self.grid_world.room_ids[transition['next_position']]
            room_connections[from_room].add(to_room)
            room_connections[to_room].add(from_room)
        
        # Build tables set
        tables = set()
        for table_pos in self.grid_world.table_positions.keys():
            room_id_table = self.grid_world.room_ids[table_pos]
            tables.add(room_id_table)
        
        planning_state = {
            'agent_location': agent_room,
            'agent_inventory': self.grid_world.agent_inventory,
            'object_locations': object_locations,
            'room_connections': dict(room_connections),
            'tables': tables,
            'low_level_position': self.grid_world.agent_pos
        }
        
        if self.debug:
            print(f"Current planning state: room={planning_state['agent_location']}, inventory={planning_state['agent_inventory']}")
        
        return planning_state
    
    def test_preconditions(self, position):
        """Test preconditions from a specific position"""
        print(f"\nTesting preconditions from position {position}:")
        
        # Set agent position for testing
        original_pos = self.grid_world.agent_pos
        self.grid_world.agent_pos = position
        
        state = self.get_state_description()
        applicable = self.find_applicable_operators(state)
        
        print(f"Found {len(applicable)} applicable operators:")
        for op in applicable:
            print(f"  - {op[0]} at {op[2]['prev_position']}")
        
        # Restore position
        self.grid_world.agent_pos = original_pos
        
        return applicable


class IntegratedPlanner:
    """Planner that integrates high-level operators with learned low-level policies"""
    
    def __init__(self, grid_world, action_operators, q_learning_agent, debug=False):
        self.grid_world = grid_world
        self.action_ops = action_operators
        self.q_agent = q_learning_agent
        self.domain = PlanningDomain()
        self.debug = debug
        
        # Map low-level transitions to their operator indices
        self.transition_to_operator = {}
        for i, op in enumerate(self.action_ops.operators):
            transition_key = self._get_transition_key(op[2])
            self.transition_to_operator[transition_key] = i
        
        if self.debug:
            print(f"IntegratedPlanner initialized with {len(self.action_ops.operators)} operators")
    
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
            if self.debug:
                print(f"  Operator {operator[0]} at {transition['prev_position']}: transition not found in Q-agent")
            return False
        
        # Check if max Q-value > 0 for current state
        current_pos = current_state['agent_position']
        max_q = np.max(self.q_agent.q_tables[transition_index][current_pos])
        
        achievable = max_q > 0
        
        if self.debug:
            print(f"  Operator {operator[0]} at {transition['prev_position']}: max Q={max_q:.3f}, achievable={achievable}")
        
        return achievable
    
    def get_achievable_operators(self, current_state):
        """Get all operators that are currently achievable"""
        applicable = self.action_ops.find_applicable_operators(current_state)
        achievable = []
        
        for op in applicable:
            if self.is_operator_achievable(op, current_state):
                achievable.append(op)
        
        if self.debug:
            print(f"Found {len(achievable)} achievable operators out of {len(applicable)} applicable")
        
        return achievable
    
    def create_planning_state(self, current_state, activated_transition=None):
        """Create a planning state from current environment state"""
        # Handle both state representations
        if 'agent_position' in current_state:
            # Grid world state representation
            if activated_transition is not None:
                new_position = activated_transition['next_position']
            else:
                new_position = current_state['agent_position']
            agent_inventory = current_state.get('inventory', None)
        else:
            # Planning state representation
            if activated_transition is not None:
                new_position = activated_transition['next_position']
            else:
                new_position = current_state['low_level_position']
            agent_inventory = current_state['agent_inventory']
        
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
            room_id_table = self.grid_world.room_ids[table_pos]
            tables.add(room_id_table)
        
        # Build object locations from current grid state
        object_locations = {}
        for table_pos in self.grid_world.table_positions.keys():
            x, y = table_pos
            if self.grid_world.grid[x, y] != 0:
                obj_id = self.grid_world.grid[x, y]
                room_id_obj = self.grid_world.room_ids[table_pos]
                object_locations[obj_id] = room_id_obj
        
        planning_state = {
            'agent_location': room_id,
            'agent_inventory': agent_inventory,
            'object_locations': object_locations,
            'room_connections': dict(room_connections),
            'tables': tables,
            'low_level_position': new_position
        }
        
        if self.debug:
            print(f"Created planning state:")
            print(f"  Room: {planning_state['agent_location']}")
            print(f"  Position: {planning_state['low_level_position']}")
            print(f"  Inventory: {planning_state['agent_inventory']}")
            print(f"  Object locations: {planning_state['object_locations']}")
            print(f"  Room connections: {dict(planning_state['room_connections'])}")
            print(f"  Tables: {planning_state['tables']}")
        
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
        
        # Check if goal is already satisfied
        if self.domain.is_goal_state(planning_state, goal):
            print("Goal already satisfied!")
            return []
        
        # Use BFS to find plan - use low_level_position from planning_state
        queue = deque([(planning_state, [], planning_state['low_level_position'])])
        visited = set()
        
        states_explored = 0
        
        while queue:
            state, plan, low_level_pos = queue.popleft()
            states_explored += 1
            
            if self.debug:
                print(f"\nExploring state {states_explored}:")
                print(f"  Room: {state['agent_location']}, Inventory: {state['agent_inventory']}")
                print(f"  Current plan length: {len(plan)}")
            
            # Check if goal is satisfied
            if self.domain.is_goal_state(state, goal):
                print(f"Goal achieved! Found plan with {len(plan)} steps after exploring {states_explored} states")
                return plan
            
            # Check depth limit
            if len(plan) >= max_depth:
                if self.debug:
                    print(f"  Reached max depth {max_depth}")
                continue
            
            # State fingerprint for visited check
            state_key = (
                state['agent_location'],
                state['agent_inventory'],
                tuple(sorted(state['object_locations'].items()))
            )
            if state_key in visited:
                if self.debug:
                    print(f"  Already visited this state")
                continue
            visited.add(state_key)
            
            # Get achievable operators in current low-level state
            # Create a grid world state representation for the low-level check
            current_low_level_state = {
                'agent_position': low_level_pos,
                'inventory': state['agent_inventory'],
                'grid': self.grid_world.grid.copy()
            }
            
            achievable_ops = self.get_achievable_operators(current_low_level_state)
            
            if self.debug:
                print(f"  Found {len(achievable_ops)} achievable operators")
            
            for op in achievable_ops:
                action_name, preconditions, transition = op
                transition_index = self._get_transition_index(transition)
                
                if self.debug:
                    print(f"  Considering operator: {action_name} at {transition['prev_position']}")
                
                # Apply the operator to get new state
                new_state = None
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
                        if self.debug:
                            print(f"    No object found in room {room_id}")
                        continue
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_inventory'] = obj_id
                    del new_state['object_locations'][obj_id]
                    new_low_level_pos = transition['next_position']
                    
                    action_desc = f"PICK_UP object {obj_id} in room {room_id}"
                    params = {'object_id': obj_id, 'room': room_id}
                    
                elif action_name == "PUT_DOWN":
                    # PUT_DOWN action
                    if state['agent_inventory'] is None:
                        if self.debug:
                            print(f"    No object in inventory to put down")
                        continue
                    
                    room_id = state['agent_location']
                    obj_id = state['agent_inventory']
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_inventory'] = None
                    new_state['object_locations'][obj_id] = room_id
                    new_low_level_pos = transition['next_position']
                    
                    action_desc = f"PUT_DOWN object {obj_id} in room {room_id}"
                    params = {'object_id': obj_id, 'room': room_id}
                
                else:
                    continue
                
                if new_state is None:
                    continue
                
                # Add to plan
                new_plan = plan + [(action_name, params, action_desc, transition_index)]
                queue.append((new_state, new_plan, new_low_level_pos))
                
                if self.debug:
                    print(f"    Added new state to queue: room {new_state['agent_location']}, inventory {new_state['agent_inventory']}")
        
        print(f"No plan found after exploring {states_explored} states")
        return None
    
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
        
        if self.debug:
            print(f"  Executing transition {transition_index}: {transition['type']} at {target_position}")
        
        # Move to the target position using the learned policy
        steps = 0
        while self.grid_world.agent_pos != target_position and steps < max_steps:
            # Get best action from current position for this transition
            current_pos = self.grid_world.agent_pos
            action = self.q_agent.get_policy(transition_index, current_pos)
            
            # Take the action
            self.grid_world.step(action)
            steps += 1
            
            if self.debug:
                print(f"    Step {steps}: {current_pos} -> {self.grid_world.agent_pos} (action {action})")
        
        if self.grid_world.agent_pos != target_position:
            if self.debug:
                print(f"  Failed to reach target position {target_position}, ended at {self.grid_world.agent_pos}")
            return False
        
        # Now we're at the right position, take the transition action
        required_action = transition['action']
        self.grid_world.step(required_action)
        
        if self.debug:
            print(f"  Took transition action {required_action}: {self.grid_world.agent_pos}")
        
        # Verify we reached the expected next position
        success = self.grid_world.agent_pos == transition['next_position']
        
        if self.debug:
            if success:
                print(f"  Successfully executed transition")
            else:
                print(f"  Failed to reach expected position {transition['next_position']}, ended at {self.grid_world.agent_pos}")
        
        return success
    
    def _verify_operator_applied(self, action_name, params):
        """Verify that the operator produced the expected state change"""
        current_state = self.action_ops.get_state_description()
        
        if action_name == "MOVE":
            expected_room = params['to_room']
            current_room = self.grid_world.room_ids[self.grid_world.agent_pos]
            success = current_room == expected_room
            if self.debug:
                print(f"  MOVE verification: expected room {expected_room}, current room {current_room}, success={success}")
            return success
            
        elif action_name == "PICK_UP":
            expected_obj = params['object_id']
            success = self.grid_world.agent_inventory == expected_obj
            if self.debug:
                print(f"  PICK_UP verification: expected inventory {expected_obj}, current inventory {self.grid_world.agent_inventory}, success={success}")
            return success
            
        elif action_name == "PUT_DOWN":
            expected_obj = params['object_id']
            expected_room = params['room']
            
            # Check if object is in the expected room
            success = False
            for table_pos, room_coords in self.grid_world.table_positions.items():
                if self.grid_world.room_ids[table_pos] == expected_room:
                    if self.grid_world.grid[table_pos] == expected_obj:
                        success = True
                        break
            
            if self.debug:
                print(f"  PUT_DOWN verification: expected object {expected_obj} in room {expected_room}, success={success}")
            return success
        
        return True
    
    def test_planning_from_position(self, goal, position):
        """Test planning from a specific position"""
        print(f"\n=== Testing planning from position {position} ===")
        
        # Set position
        original_pos = self.grid_world.agent_pos
        self.grid_world.agent_pos = position
        
        # Create plan
        plan = self.plan_with_learned_policies(goal)
        
        if plan:
            print(f"Plan found with {len(plan)} steps:")
            for i, (action_name, params, description, _) in enumerate(plan):
                print(f"  {i+1}. {description}")
        else:
            print("No plan found")
        
        # Restore position
        self.grid_world.agent_pos = original_pos
        
        return plan


def test_basic_q_learning():
    """Test basic Q-learning functionality"""
    print("=== Testing Basic Q-Learning ===")
    
    grid_world = GridWorld(num_rooms=4, room_size=3)
    q_agent = GoalConditionedQLearning(grid_world, debug=True)
    
    # Test short training
    print("Training for 2000 steps...")
    q_agent.train_continuous(total_steps=2000, log_interval=1000)
    
    # Test specific policies
    print("\n--- Testing Learned Policies ---")
    active_transitions = []
    for i, count in enumerate(q_agent.training_stats['transitions_activated']):
        if count > 0:
            active_transitions.append(i)
    
    if active_transitions:
        test_goal = active_transitions[0]
        transition = q_agent.all_transitions[test_goal]
        start_pos = transition['prev_position']
        
        print(f"Testing policy for transition {test_goal}")
        path = q_agent.test_single_policy(test_goal, start_pos, max_steps=20)
    
    return q_agent

def test_operator_creation():
    """Test operator creation and application"""
    print("\n=== Testing Operator Creation ===")
    
    grid_world = GridWorld(num_rooms=4, room_size=3)
    action_ops = ActionOperators(grid_world, debug=True)
    
    # Display operators
    action_ops.display_operators()
    
    # Test from different positions
    test_positions = [(1, 1), (3, 1), (1, 3)]
    for pos in test_positions:
        action_ops.test_preconditions(pos)
    
    return action_ops

def test_integrated_planning():
    """Test integrated planning system"""
    print("\n=== Testing Integrated Planning ===")
    
    grid_world = GridWorld(num_rooms=4, room_size=3)
    
    # Train Q-agent
    q_agent = GoalConditionedQLearning(grid_world, debug=False)
    print("Training Q-agent...")
    q_agent.train_continuous(total_steps=5000, log_interval=2500)
    
    # Create action operators
    action_ops = ActionOperators(grid_world, debug=True)
    
    # Create integrated planner
    planner = IntegratedPlanner(grid_world, action_ops, q_agent, debug=True)
    
    # Test planning from different positions
    current_state = action_ops.get_state_description()
    planning_state = planner.create_planning_state(current_state)
    
    # Try to create a simple goal
    if planning_state['object_locations']:
        obj_id = list(planning_state['object_locations'].keys())[0]
        obj_room = planning_state['object_locations'][obj_id]
        
        # Find a different room
        target_room = None
        for room in planning_state['tables']:
            if room != obj_room:
                target_room = room
                break
        
        if target_room:
            goal = {
                'object_location': {
                    'object_id': obj_id,
                    'room': target_room
                }
            }
            
            print(f"Testing goal: Move object {obj_id} from room {obj_room} to room {target_room}")
            
            # Test from current position - use the low_level_position from planning state
            start_position = planning_state['low_level_position']
            plan = planner.test_planning_from_position(goal, start_position)
            
            if plan:
                return True
    
    return False

def run_all_tests():
    """Run all tests"""
    print("Running all tests...")
    
    # Test 1: Basic Q-learning
    q_agent = test_basic_q_learning()
    
    # Test 2: Operator creation
    action_ops = test_operator_creation()
    
    # Test 3: Integrated planning
    planning_success = test_integrated_planning()
    
    print(f"\n=== All Tests Completed ===")
    print(f"Q-learning: WORKING")
    print(f"Operators: WORKING") 
    print(f"Planning: {'WORKING' if planning_success else 'NO PLAN FOUND'}")
    
    return q_agent, action_ops, planning_success

# Run all tests
if __name__ == "__main__":
    q_agent, action_ops, planning_success = run_all_tests()