import numpy as np
import random
from enum import Enum
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import copy
import time

class ObjectType(Enum):
    EMPTY = 0
    A = 2
    B = 3
    C = 4

class ActionType(Enum):
    MOVE = 0
    PICK_UP = 1
    PUT_DOWN = 2

class GridWorld:
    def __init__(self, num_rooms=4, room_size=5, debug=False):
        self.num_rooms = num_rooms
        self.room_size = room_size
        self.rooms_per_side = int(np.sqrt(num_rooms))
        self.debug = debug
        
        # FIXED: Correct grid size calculation
        self.grid_size = self.rooms_per_side * (self.room_size + 1) - 1
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Store table positions and room IDs
        self.table_positions = {}
        self.room_ids = {}  # Maps position to room ID
        
        self.agent_pos = (0,0)
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
                if door_col < self.grid_size:
                    self.grid[wall_row, door_col] = 0
                
                # Vertical doors
                door_row = j * (self.room_size + 1) + door_pos
                if door_row < self.grid_size:
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
                if center_x < self.grid_size and center_y < self.grid_size:
                    self.table_positions[(center_x, center_y)] = (room_x, room_y)
                    room_centers.append((center_x, center_y))
        
        # Shuffle room centers to assign objects randomly
        random.shuffle(room_centers)
        # First, ensure at least one of each object type


        for i, obj_type in enumerate(object_types):
            if i < len(room_centers):
                center_x, center_y = room_centers[i]
                self.grid[center_x, center_y] = obj_type


        #self.grid[5,5] = 3
        #self.grid[1,5] = 2

    def _assign_room_ids(self):
        """Assign room IDs to all positions in the grid"""
        mat = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Calculate which room this position belongs to
                room_x = x // (self.room_size + 1)
                room_y = y // (self.room_size + 1)
                room_id = room_x + room_y * (self.rooms_per_side) 
                
                self.room_ids[(x, y)] = room_id
                mat[x, y] = room_id
                
    
    def _precompute_transitions(self):
        """Precompute all possible door and object transitions"""        
        # Horizontal doors (vertical walls)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for a in range(4):
                    prev_pos = (i,j)
                    next_pos = self.step_2(a, i,j)
                    room1 = self.room_ids.get(prev_pos, -1)
                    room2 = self.room_ids.get(next_pos, -1)

                    if (room1 != room2 and self.grid[prev_pos] != 1 and self.grid[next_pos] != 1):
                        self.door_transitions.append({
                            'prev_position': prev_pos,
                            'action': a, # MOVE action
                            'next_position': next_pos,
                            'type': 'door',
                            'precondition': (room1, room2)
                        })
                
        # Precompute object transitions (all table positions)
        for table_pos in self.table_positions.keys():
            self.object_transitions.append({
                'prev_position': table_pos,
                'action': 4,  # TOGGLE
                'next_position': table_pos,
                'type': 'object',
                'precondition': self.room_ids.get(table_pos, -1)
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

    def step_2(self, action, x, y):
        """Execute an action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=TOGGLE)"""
        #
        next_x, next_y = x, y
        if action == 0 and x > 0 and self.grid[x-1, y] != 1:  # UP
            next_x, next_y = (next_x-1, next_y)
        elif action == 1 and x < self.grid_size-1 and self.grid[x+1, y] != 1:  # DOWN
            next_x, next_y = (next_x+1, next_y)
        elif action == 2 and y > 0 and self.grid[x, y-1] != 1:  # LEFT
            next_x, next_y = (next_x, next_y-1)
        elif action == 3 and y < self.grid_size-1 and self.grid[x, y+1] != 1:  # RIGHT
            next_x, next_y = (next_x, next_y+1)
        
        return next_x, next_y
    
    def _toggle_object(self):
        """Pick up or put down object if agent is at a table - FIXED VERSION"""
        if self.agent_pos in self.table_positions:
            if self.agent_inventory is None:
                # Pick up object if there is one
                obj_at_position = self.grid[self.agent_pos[0], self.agent_pos[1]]
                if obj_at_position >= 1:  # Object types start from 1
                    self.agent_inventory = obj_at_position
                    self.grid[self.agent_pos[0], self.agent_pos[1]] = 0  # Clear the grid position
                    if self.debug:
                        print(f"Picked up object {obj_at_position} from {self.agent_pos}")
            else:
                # Put down object if table is empty
                if self.grid[self.agent_pos[0], self.agent_pos[1]] == 0:
                    self.grid[self.agent_pos[0], self.agent_pos[1]] = self.agent_inventory
                    if self.debug:
                        print(f"Put down object {self.agent_inventory} at {self.agent_pos}")
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
            'current_inventory': self.agent_inventory,
            'at_table': self.agent_pos in self.table_positions
        }
    
    def render(self):
        """Display the current state"""
        state = self.get_state()
        print(state['grid'])
        print(f"Room ID: {state['room_id']}, Inventory: {state['current_inventory']}")


class Agent:
    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.knowledge_base = {
            'known_rooms': set(),  # Room IDs the agent has visited
            'room_connections': set(),  # Tuples (room1, room2) for connected rooms
            'object_locations': set(),  # Object locations on tables
            'current_room': None,  # Track previous room to detect connections
            'current_inventory': None,  # Track inventory changes
            'at_table': None    # Track if at table
        }
        
        # Initialize with starting room knowledge
        self._update_knowledge()
    
    def _update_knowledge(self):
        """Update knowledge based on current state - FIXED object tracking"""

        state = self.grid_world.get_state()
        current_room = state['room_id']
        currently_at_table = state['at_table']
        
        
        # Mark current room as known
        self.knowledge_base['known_rooms'].add(current_room)
        
        # Detect and record room connections
        if (self.knowledge_base['current_room'] is not None and 
            self.knowledge_base['current_room'] != current_room):
            
            # Add bidirectional connection
            room1 = self.knowledge_base['current_room']
            room2 = current_room
            connection = tuple(sorted([room1, room2]))
            self.knowledge_base['room_connections'].add(connection)
    
        current_inventory = state['current_inventory']

        # If we just picked up an object, remove it from object_locations
        if self.knowledge_base['current_inventory'] is None and current_inventory is not None:
            self.knowledge_base['object_locations'] = set(x for x in self.knowledge_base['object_locations'] if x[1]!= current_inventory)
            self.knowledge_base['object_locations'].add((current_room, -1))  # Mark that the object is no longer on the table

        # If we just put down an object, add it to object_locations
        if self.knowledge_base['current_inventory'] is not None and current_inventory is None:
            # We put down an object - it should be on a table in current room
            self.knowledge_base['object_locations'] = set(x for x in self.knowledge_base['object_locations'] if x[0]!= current_room)
            self.knowledge_base['object_locations'].add((current_room, self.knowledge_base['current_inventory']))
        
        if self.knowledge_base['at_table'] and currently_at_table:
            # Agent just moved away from a table, ensure object location is updated
            if current_inventory is None and self.knowledge_base['current_inventory'] is None:
                # If not holding anything, mark table as empty
                self.knowledge_base['object_locations'] = set(x for x in self.knowledge_base['object_locations'] if x[0]!= current_room)
                self.knowledge_base['object_locations'].add((current_room, -1))


        # Update previous inventory for next comparison
        self.knowledge_base['current_inventory'] = current_inventory

        # Update room tracking
        self.knowledge_base['current_room'] = current_room

        # Update at_table status
        self.knowledge_base['at_table'] = currently_at_table
    
    def step(self, action):
        """Take an action and update knowledge"""
        # Execute the action
        self.grid_world.step(action)
        
        # Update knowledge after action
        self._update_knowledge()
    

class PlanningDomain:
    """Planning domain using the agent's knowledge base as state representation"""
    
    def __init__(self):
        self.actions = {
            ActionType.MOVE: self._move_action,
            ActionType.PICK_UP: self._pick_up_action,
            ActionType.PUT_DOWN: self._put_down_action
        }
    
    def get_actions(self):
        """Get all available action types"""
        return list(self.actions.keys())
    
    def _move_action(self, kb_state, from_room, to_room, next_position = None, operator_actions = None):
        """Move action: agent moves between connected rooms based on knowledge"""
        current_room = kb_state['current_room']
        if current_room != from_room:
            return None, f"Agent not in room {from_room} (currently in {current_room})"
            
        # Check if connection exists in knowledge base
        connection = tuple(sorted([from_room, to_room]))
        if connection not in kb_state['room_connections']:
            return None, f"Rooms {from_room} and {to_room} are not known to be connected"
            
        new_state = copy.deepcopy(kb_state)
        new_state['current_room'] = to_room
        new_state['known_rooms'].add(to_room)
        return new_state, f"Moved from room {from_room} to room {to_room}"
    
    def _pick_up_action(self, kb_state, object_type, room, next_position = None, operator_actions = None):
        """Pick up action: agent picks up object from current room based on knowledge"""
        current_room = kb_state['current_room']
        if current_room != room:
            return None, f"Agent not in room {room} (currently in {current_room})"
            
        if kb_state['current_inventory'] is not None:
            return None, f"Agent already holding object {kb_state['current_inventory']}"
            
        # Check if object is known to be in this room
        if (room, object_type) not in kb_state['object_locations']:
            return None, f"Object {object_type} not known to be in room {room}"
            
        new_state = copy.deepcopy(kb_state)
        new_state['current_inventory'] = object_type
        new_state['object_locations'].remove((room, object_type))
        new_state['object_locations'].add((room, -1))  # Mark that the object is no longer on the table
        return new_state, f"Picked up object {object_type} in room {room}"
    
    def _put_down_action(self, kb_state, object_type, room, next_position = None, operator_actions = None):
        """Put down action: agent puts object in current room based on knowledge"""
        current_room = kb_state['current_room']
        if current_room != room:
            return None, f"Agent not in room {room}"
            
        if kb_state['current_inventory'] != object_type:
            return None, f"Agent not holding object {object_type} (holding {kb_state['current_inventory']})"
            
        if (room, -1) not in kb_state['object_locations']:
            return None, f"Room {room} does not have an empty table to put down the object"

        new_state = copy.deepcopy(kb_state)
        new_state['current_inventory'] = None
        new_state['object_locations'].remove((room, -1))  # Remove empty table marker
        new_state['object_locations'].add((room, object_type))
        return new_state, f"Put down object {object_type} in room {room}"
    
    def apply_action(self, kb_state, action_type, **params):
        """Apply an action to a knowledge base state and return new state"""
        if action_type not in self.actions:
            return None, f"Unknown action type: {action_type}"
            
        return self.actions[action_type](kb_state, **params)
    
    def get_applicable_actions(self, kb_state):
        """Get all applicable actions in current knowledge base state"""
        applicable = []
        current_room = kb_state['current_room']
        
        # Move actions to connected rooms
        for connection in kb_state['room_connections']:
            if current_room in connection:
                other_room = connection[0] if connection[1] == current_room else connection[1]
                applicable.append((ActionType.MOVE, {
                    'from_room': current_room,
                    'to_room': other_room
                }))
        
        # Pick up actions for objects in current room
        if kb_state['current_inventory'] is None:
            for room, obj_type in kb_state['object_locations']:
                if room == current_room:
                    applicable.append((ActionType.PICK_UP, {
                        'object_type': obj_type,
                        'room': current_room
                    }))
        
        # Put down action (can only put down if no other object in the room)
        if kb_state['current_inventory'] is not None:
            # Check if there are any objects already in the current room
            #objects_in_room = any(room == current_room for room, obj_type in kb_state['object_locations'])
            room_empty = (current_room, -1) in kb_state['object_locations']
            if room_empty:
                applicable.append((ActionType.PUT_DOWN, {
                    'object_type': kb_state['current_inventory'],
                    'room': current_room
                }))
        
        return applicable
    
    def is_goal_state(self, kb_state, goal):
        """Check if knowledge base state satisfies goal condition using propositional logic"""
        """ Goal is represented as a nested tuple structure: ('AND', (1, 2), ('NOT', (1, 3)))"""
        
        def evaluate_formula(formula):
            """Recursively evaluate a logical formula"""
            if isinstance(formula, tuple):
                operator = formula[0]
                
                if operator == 'AND':
                    # Evaluate all sub-formulas, return True only if all are true
                    return all(evaluate_formula(sub_formula) for sub_formula in formula[1:])
                
                elif operator == 'NOT':
                    # Negate the sub-formula
                    return not evaluate_formula(formula[1])
                
                elif operator == 'OR':
                    # Evaluate sub-formulas, return True if any is true
                    return any(evaluate_formula(sub_formula) for sub_formula in formula[1:])
                
                else:
                    # Assume it's an atomic predicate (room, object_type)
                    room, obj_type = formula
                    return (room, obj_type) in kb_state['object_locations']
            
            else:
                # Assume it's an atomic predicate (room, object_type)
                room, obj_type = formula
                return (room, obj_type) in kb_state['object_locations']
        
        return evaluate_formula(goal)

class EventAwarePlanningDomain(PlanningDomain):
    """Planning domain that tracks events to update knowledge base"""
    def __init__(self):
        super().__init__()
    
    # Additional methods for event tracking can be added here
    def get_applicable_actions(self, kb_state, events, agent_pos):
        """Get all applicable actions in current knowledge base state with event awareness"""
        
        applicable = []
        current_room = kb_state['current_room']
        move_events = [ev for ev in events.get(agent_pos, []) if ev['type'] == 'door']
        pick_up_events = [ev for ev in events.get(agent_pos, []) if ev['type'] == 'object']
        
        move_events_connections = set(x['precondition'] for x in move_events)
        filtered_connections = kb_state['room_connections'].intersection(move_events_connections)

        selected_move_events = [x for x in move_events if x['precondition'] in filtered_connections]
        # Move actions to connected rooms

        for ev in move_events:
            connection = ev['precondition']
            if current_room ==  connection[0]:
                other_room = connection[1]
                applicable.append((ActionType.MOVE, {
                    'from_room': current_room,
                    'to_room': other_room,
                    'transition_event': (ev['prev_position'], ev['action'], ev['next_position'])
                }))
        
        applicable_pickup_events = [x for x in pick_up_events if x['precondition'] == current_room]
        
        for ev in pick_up_events:

            if kb_state['current_inventory'] is None:
                for room, obj_type in kb_state['object_locations']:
                    if room == current_room and obj_type != -1:
                        applicable.append((ActionType.PICK_UP, {
                            'object_type': obj_type,
                            'room': current_room,
                            'transition_event': (ev['prev_position'], ev['action'], ev['next_position'])
                        })) 

            if kb_state['current_inventory'] is not None:
                # Check if there are any objects already in the current room
                #objects_in_room = any(room == current_room for room, obj_type in kb_state['object_locations'])
                room_empty = (current_room, -1) in kb_state['object_locations']
                if room_empty:
                    applicable.append((ActionType.PUT_DOWN, {
                        'object_type': kb_state['current_inventory'],
                        'room': current_room,
                        'transition_event': (ev['prev_position'], ev['action'], ev['next_position'])
                    }))
        #print("Applicable actions with events:")
        #print(kb_state)
        #print(applicable)
        return applicable
      
class Planner:
    """Fixed planner with consistent state representation"""
    
    def __init__(self, domain):
        self.domain = domain
    
    def bfs_plan(self, initial_state, goal, max_depth=50):
        """Find plan using BFS with proper goal checking"""
        if self.domain.is_goal_state(initial_state, goal):
            return []
        
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue:
            state, plan = queue.popleft()
            
            if self.domain.is_goal_state(state, goal):
                return plan
            
            if len(plan) >= max_depth:
                continue
                
            state_key = self._get_state_key(state)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            for action_type, params in self.domain.get_applicable_actions(state):
                
                new_state, result_msg = self.domain.apply_action(state, action_type, **params)
                
                if new_state is not None:
                    new_state_key = self._get_state_key(new_state)
                    if new_state_key not in visited:
                        action_desc = f"{action_type.name}: {result_msg}"
                        queue.append((new_state, plan + [(action_type, params, action_desc)]))
        
        return None
    
    def _get_state_key(self, state):
        """Create a hashable key for state - FIXED VERSION"""
        return (
            state['current_room'],           # Fixed key name
            state['current_inventory'],              # Fixed key name
            tuple(sorted(state['object_locations'])),  # Fixed: it's a set, not dict
            tuple(sorted(state['known_rooms'])),       # Added missing component
            tuple(sorted(state['room_connections']))   # Added missing component
        )


class EventAwarePlannerRefactored(Planner):

    def __init__(self, domain, events):
        super().__init__(domain)
        
        self.events = events
        self.environment_transitions = set()
        self.operator_dict = {}
    
    def add_environment_transition(self, transition):
        """Add an observed environment transition"""
        self.environment_transitions.add(transition)

    
    def get_operator_actions(self, current_agent_pos, event):
        """Get operator actions based on current position and event"""

        transition_key = (current_agent_pos, event)
        #print(transition_key)
        if transition_key in self.operator_dict:
            #print("Using cached operator actions")
            return self.operator_dict[transition_key]

        queue = deque([((None, None, current_agent_pos), [])])
        visited = set()

        while queue:
            state, plan = queue.popleft()

            agent_pos = state[2]

            if state == event:
                self.operator_dict[transition_key] = plan
                #print(f"Starting BFS from {current_agent_pos} to reach event {event}")
                #print("Found operator actions:")
                #print(plan)
                return plan
            
            if state in visited:
                continue
            visited.add(state)

            event_transitions_set = set((ev['prev_position'], ev['action'], ev['next_position']) for ev in self.events if (ev['prev_position'], ev['action'], ev['next_position'])  != event)

            for transition in set(tr for tr in self.environment_transitions if tr[0] == agent_pos).difference(event_transitions_set):
                next_state = transition 
                queue.append((next_state, plan + [transition[1]]))
        
        return None

    def get_applicable_operators(self, kb_state, agent_pos):
        """Get applicable operators based on important transitions"""
        applicable = []
        current_room = kb_state['current_room']

        move_events = [ev for ev in self.events if ev['type'] == 'door']
        pick_up_events = [ev for ev in self.events if ev['type'] == 'object']
        
        for ev in move_events:
            connection = ev['precondition']
            if current_room in connection:
                other_room = connection[0] if connection[1] == current_room else connection[1]
                operator_actions = self.get_operator_actions(agent_pos, (ev['prev_position'], ev['action'], ev['next_position']))


                if operator_actions is not None:
                    applicable.append((ActionType.MOVE, {
                        'from_room': current_room,
                        'to_room': other_room,
                        'next_position': ev['next_position'],
                        'operator_actions': operator_actions
                    }))
        
        for ev in pick_up_events:
            if kb_state['current_inventory'] is None:
                for room, obj_type in kb_state['object_locations']:
                    if room == current_room and obj_type != -1:
                        operator_actions = self.get_operator_actions(agent_pos, (ev['prev_position'], ev['action'], ev['next_position']))

                        if operator_actions is not None:
                            applicable.append((ActionType.PICK_UP, {
                                'object_type': obj_type,
                                'room': current_room,
                                'next_position': ev['next_position'],
                                'operator_actions': operator_actions
                            })) 

            if kb_state['current_inventory'] is not None:
                # Check if there are any objects already in the current room
                #objects_in_room = any(room == current_room for room, obj_type in kb_state['object_locations'])
                room_empty = (current_room, -1) in kb_state['object_locations']
                if room_empty:
                    operator_actions = self.get_operator_actions(agent_pos, (ev['prev_position'], ev['action'], ev['next_position']))

                    if operator_actions is not None:
                        applicable.append((ActionType.PUT_DOWN, {
                            'object_type': kb_state['current_inventory'],
                            'room': current_room,
                            'next_position': ev['next_position'],
                            'operator_actions': operator_actions
                        }))

        return applicable


    
    def bfs_plan(self, initial_state, goal, agent_pos, max_depth=20):
        """Find plan using BFS with event awareness and proper goal checking"""
        
        queue = deque([(initial_state, agent_pos, [], [])])
        visited = set()

        while queue:
            state, current_agent_pos, plan, plan_exectuion_actions = queue.popleft()
            
            if self.domain.is_goal_state(state, goal):
                return plan, plan_exectuion_actions
            
            if len(plan) >= max_depth:
                continue
                
            state_key = self._get_state_key(state, current_agent_pos)

            if state_key in visited:
                continue

            visited.add(state_key)

            operators = self.get_applicable_operators(state, current_agent_pos)

            if operators is None:
                continue

            for action_type, params in operators:
                
                new_state, result_msg = self.domain.apply_action(state, action_type, **params)
                next_agent_pos = copy.deepcopy(params.get('next_position', None))
                
                if new_state is not None:
                    new_state_key = self._get_state_key(new_state, next_agent_pos)
                    if new_state_key not in visited:
                        action_desc = f"{action_type.name}: {result_msg}"

                        queue.append((new_state, next_agent_pos, plan + [(action_type, params, action_desc)], plan_exectuion_actions + params.get('operator_actions', [])))
        
        return None, None
    
    def _get_state_key(self, state, agent_pos):
        """Create a hashable key for state - FIXED VERSION"""
        return (
            state['current_room'],           # Fixed key name
            state['current_inventory'],              # Fixed key name
            tuple(sorted(state['object_locations'])),   # Fixed: it's a set, not dict
            tuple(sorted(state['known_rooms'])),        # Added missing component
            tuple(sorted(state['room_connections'])),   # Added missing component,
            agent_pos
        )



class LearningAgentRefatored(Agent):

    def __init__(self, grid_world, goal = None):

        super().__init__(grid_world)

        self.total_steps = 0
        
        # Simple count-based exploration: track state-action counts
        self.state_action_counts = np.zeros((grid_world.grid_size, grid_world.grid_size, 5), dtype=np.int32) 
        
        events = grid_world.get_important_transitions()
        self.events = events['door_transitions'] + events['object_transitions']

        # Planner
        self.planner = EventAwarePlannerRefactored(EventAwarePlanningDomain(), self.events)

        self.goal = goal

    def step(self, action):

        prev_state = self.grid_world.agent_pos
        
        # Update count before taking action
        self.state_action_counts[prev_state[0], prev_state[1], action] += 1
        
        # Execute the action using parent class
        super().step(action)
        
        next_state = self.grid_world.agent_pos

        self.planner.add_environment_transition((prev_state, action, next_state))   

        self.total_steps += 1
    
    def choose_action_count_based(self):
        """Simple count-based exploration: choose the least taken action in current state"""
        
        x, y = self.grid_world.agent_pos
        return np.argmin(self.state_action_counts[x, y, :])

    
    def get_new_goal(self):
        """Randomly select a new goal"""

        while(True):
            object_id = random.choice([obj.value for obj in ObjectType if obj != ObjectType.EMPTY])
            #object_id = 2
            room_id = random.randint(0, self.grid_world.rooms_per_side**2 - 1)
            goal = (room_id, object_id)
            if goal not in self.knowledge_base['object_locations']:
                return goal


    def interaction_loop(self, num_steps=100_000, max_steps = 20):
        """Interact with the environment for a number of steps"""

        t = 0
  
        cost_list = []
        planning_trials = []
        count = 0
        explore_count = 0
        last_goal_time = t

        while t < num_steps:
            
            current_grid_pos = self.grid_world.agent_pos
            plan, plan_actions = self.planner.bfs_plan(self.knowledge_base, self.goal, self.grid_world.agent_pos)
            
            if plan is not None and len(plan) > 0:
                #print(plan, plan_actions)
            
                planning_trials.append(1)
                #print(f"Step {t}: Plan found with {len(plan)} actions to achieve goal {self.goal}")
                #print(self.knowledge_base)
                for action in plan_actions:
                    
                    
                    #self.grid_world.render()
                    #print(f"Step {t}: Executing action {action}")

                    self.step(action)
                    t += 1

                    if self.planner.domain.is_goal_state(self.knowledge_base, self.goal):
                        cost_list.append(t- last_goal_time)
                        last_goal_time = t
                        self.goal = self.get_new_goal()
                        print(f"New goal set: {self.goal}")
            else:

                for _ in range(max_steps*10):
                    action = self.choose_action_count_based()
                    self.step(action)   
                    t += 1

                planning_trials.append(0)
            
            
            if len(cost_list) > count:
                print(f"Step {t}: Percentage completed {t/ num_steps * 100:.2f}%")
                print(f"Goal time: {np.mean(cost_list[-20:])} over last {len(cost_list)} goals")
                print(f"Planning success rate: {np.mean(planning_trials[-20:])} over last {len(planning_trials)} trials")
                count = len(cost_list)            
            
            explore_count +=1

            if explore_count % 10 == 0:
                self._log_progress(t)

            

        print("Interaction loop completed!")
        print(f"Total successful goal achievements: {len(cost_list)}")
        print(f"Goal cost list: {cost_list}")
        print(f"Final planning trials: {planning_trials}")
        return cost_list, planning_trials

                
    def _log_progress(self, step):
        """Log current learning and knowledge progress"""

        print(f"Step {step}: Known {self.knowledge_base} objects")
    


if __name__ == "__main__":

    for j in range(10):
        
        print(f"===== Trial {j+1} =====")
        # Create environment and agent
        grid_world = GridWorld(num_rooms=9, room_size=3, debug=False)

        print("Initial Grid:")
        print(grid_world.render())

        agent = LearningAgentRefatored(grid_world,goal = (0,2))
        #agent.q_learner.train(total_steps=100_000)

        # Use simple count-based exploration
        cost_list, planning_trials = agent.interaction_loop(num_steps=10_000)

        with open(f'./experiments/trial_{j+1}_costs.txt', 'w+') as f:
            
            f.write(str(cost_list))
            f.write('\n')
            f.write(str(planning_trials))