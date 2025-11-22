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

class ActionType(Enum):
    MOVE = 0
    PICK_UP = 1
    PUT_DOWN = 2

class GridWorld:
    def __init__(self, num_rooms=4, room_size=3, debug=False):
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
        print(mat)
                
    
    def _precompute_transitions(self):
        """Precompute all possible door and object transitions"""        
        # Horizontal doors (vertical walls)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for a in range(4):
                    prev_pos = (i,j)
                    next_pos = self.step_2(a, i,j)

                    if (self.room_ids.get(prev_pos, -1) != self.room_ids.get(next_pos, -2)):
                        self.door_transitions.append({
                            'prev_position': prev_pos,
                            'action': a,  # RIGHT
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
            'object_locations': set(),  # Object locations on tables
            'previous_room': None,  # Track previous room to detect connections
            'previous_inventory': None,  # Track inventory changes
        }
        
        # Initialize with starting room knowledge
        self._update_knowledge()
    
    def _update_knowledge(self):
        """Update knowledge based on current state - FIXED object tracking"""

        state = self.grid_world.get_state()
        current_room = state['room_id']
        
        
        # Update room tracking
        self.knowledge_base['previous_room'] = current_room
        
        # Mark current room as known
        self.knowledge_base['known_rooms'].add(current_room)
        
        # Detect and record room connections
        if (self.knowledge_base['previous_room'] is not None and 
            self.knowledge_base['previous_room'] != current_room):
            
            # Add bidirectional connection
            room1 = self.knowledge_base['previous_room']
            room2 = current_room
            connection = tuple(sorted([room1, room2]))
            self.knowledge_base['room_connections'].add(connection)
    
        current_inventory = state['inventory']

        # If we just picked up an object, remove it from object_locations
        if self.knowledge_base['previous_inventory'] is None and current_inventory is not None:
            self.knowledge_base['object_locations'] = set(x for x in self.knowledge_base['object_locations'] if x[1]!= current_inventory)

        # If we just put down an object, add it to object_locations
        elif self.knowledge_base['previous_inventory'] is not None and current_inventory is None:
            # We put down an object - it should be on a table in current room
            self.knowledge_base['object_locations'].add((current_room, self.knowledge_base['previous_inventory']))
        
        # Update previous inventory for next comparison
        self.knowledge_base['previous_inventory'] = current_inventory
    
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
    
    def _move_action(self, kb_state, from_room, to_room):
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
    
    def _pick_up_action(self, kb_state, object_type, room):
        """Pick up action: agent picks up object from current room based on knowledge"""
        current_room = kb_state['current_room']
        if current_room != room:
            return None, f"Agent not in room {room} (currently in {current_room})"
            
        if kb_state['inventory'] is not None:
            return None, f"Agent already holding object {kb_state['inventory']}"
            
        # Check if object is known to be in this room
        if (room, object_type) not in kb_state['object_locations']:
            return None, f"Object {object_type} not known to be in room {room}"
            
        new_state = copy.deepcopy(kb_state)
        new_state['inventory'] = object_type
        new_state['object_locations'].remove((room, object_type))
        return new_state, f"Picked up object {object_type} in room {room}"
    
    def _put_down_action(self, kb_state, object_type, room):
        """Put down action: agent puts object in current room based on knowledge"""
        current_room = kb_state['current_room']
        if current_room != room:
            return None, f"Agent not in room {room}"
            
        if kb_state['inventory'] != object_type:
            return None, f"Agent not holding object {object_type} (holding {kb_state['inventory']})"
            
        new_state = copy.deepcopy(kb_state)
        new_state['inventory'] = None
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
        if kb_state['inventory'] is None:
            for room, obj_type in kb_state['object_locations']:
                if room == current_room:
                    applicable.append((ActionType.PICK_UP, {
                        'object_type': obj_type,
                        'room': current_room
                    }))
        
        # Put down action (can put down in any room)
        if kb_state['inventory'] is not None:
            applicable.append((ActionType.PUT_DOWN, {
                'object_type': kb_state['inventory'],
                'room': current_room
            }))
        
        return applicable
    
    def is_goal_state(self, kb_state, goal):
        """Check if knowledge base state satisfies goal condition"""
        if 'object_location' in goal:
            obj_type = goal['object_location']['object_type']
            target_room = goal['object_location']['room']
            return (target_room, obj_type) in kb_state['object_locations']
        
        if 'inventory_contains' in goal:
            return kb_state['inventory'] == goal['inventory_contains']
            
        return False
    
    def convert_to_kb_state(self, agent):
        """Convert agent's knowledge base to planning state representation"""
        current_state = agent.grid_world.get_state()
        
        return {
            'current_room': current_state['room_id'],
            'inventory': current_state['inventory'],
            'known_rooms': copy.deepcopy(agent.knowledge_base['known_rooms']),
            'room_connections': copy.deepcopy(agent.knowledge_base['room_connections']),
            'object_locations': copy.deepcopy(agent.knowledge_base['object_locations'])
        }

class Planner:
    """Fixed planner with better goal checking and plan validation"""
    
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
        """Create a hashable key for state"""
        return (
            state['agent_location'],
            state['agent_inventory'],
            tuple(sorted(state['object_locations'].items()))
        )


class GoalConditionedQLearning:
    def __init__(self, grid_world, learning_rate=0.1, discount_factor=0.9, debug=False):
        self.grid_world = grid_world
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.debug = debug
        
        transitions = grid_world.get_important_transitions()
        self.all_transitions = transitions['door_transitions'] + transitions['object_transitions']
        
        self.q_tables = {}
        for i, transition in enumerate(self.all_transitions):
            self.q_tables[i] = defaultdict(lambda: np.zeros(5))
        
        self.q_value_sums = [[] for _ in range(len(self.all_transitions))]
        
        self.training_stats = {
            'steps': 0,
            'transitions_activated': [0] * len(self.all_transitions),
            'q_updates': 0,
            'valid_moves': 0,
            'wall_collisions': 0
        }
    
    def get_state_key(self, position):
        return position
    
    def choose_action(self, state_key, goal_index=None):
        if random.random() < 0.9:
            return random.randint(0, 4)
        else:
            if goal_index is not None:
                q_values = self.q_tables[goal_index][state_key]
                if np.max(q_values) > 0:
                    return np.argmax(q_values)
            return random.randint(0, 4)
    
    def check_transition_activation(self, prev_state, action, next_state):
        for i, transition in enumerate(self.all_transitions):
            if (transition['prev_position'] == prev_state and 
                transition['action'] == action and
                transition['next_position'] == next_state):
                
                if self.debug:
                    print(f"  Transition {i} activated: {transition['type']} at {prev_state} -> {next_state}")
                self.training_stats['transitions_activated'][i] += 1
                return i
        
        if action == 4:
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
        prev_key = self.get_state_key(prev_state)
        next_key = self.get_state_key(next_state)
        
        for goal_index in range(len(self.all_transitions)):
            if activated_transition == goal_index:
                reward = 10.0
                terminal = True
            elif activated_transition is not None:
                reward = -1.0
                terminal = True
            else:
                reward = -0.1
                terminal = False
            
            current_q = self.q_tables[goal_index][prev_key][action]
            
            if terminal:
                target = reward
            else:
                next_max = np.max(self.q_tables[goal_index][next_key])
                target = reward + self.gamma * next_max
            
            new_q = current_q + self.alpha * (target - current_q)
            self.q_tables[goal_index][prev_key][action] = new_q
            self.training_stats['q_updates'] += 1
    
    def get_valid_actions(self, position):
        x, y = position
        valid_actions = []
        
        if x > 0 and self.grid_world.grid[x-1, y] != 1:
            valid_actions.append(0)
        if x < self.grid_world.grid_size-1 and self.grid_world.grid[x+1, y] != 1:
            valid_actions.append(1)
        if y > 0 and self.grid_world.grid[x, y-1] != 1:
            valid_actions.append(2)
        if y < self.grid_world.grid_size-1 and self.grid_world.grid[x, y+1] != 1:
            valid_actions.append(3)
        valid_actions.append(4)
        
        return valid_actions
    
    def get_random_valid_position(self):
        while True:
            x = random.randint(0, self.grid_world.grid_size - 1)
            y = random.randint(0, self.grid_world.grid_size - 1)
            if self.grid_world.grid[x, y] != 1:
                return (x, y)
    
    def train_continuous(self, total_steps=100000, log_interval=10000):
        print(f"Training {len(self.all_transitions)} goal-conditioned policies for {total_steps} steps...")
        
        self.grid_world.agent_pos = self.get_random_valid_position()
        self.grid_world.agent_inventory = None
        
        prev_state = self.grid_world.agent_pos
        
        for step in range(total_steps):
            self.training_stats['steps'] += 1
            
            valid_actions = self.get_valid_actions(prev_state)
            if not valid_actions:
                self.grid_world.agent_pos = self.get_random_valid_position()
                prev_state = self.grid_world.agent_pos
                continue
                
            action = random.choice(valid_actions)
            
            self.grid_world.step(action)
            next_state = self.grid_world.agent_pos
            
            if prev_state != next_state:
                self.training_stats['valid_moves'] += 1
            else:
                self.training_stats['wall_collisions'] += 1
            
            activated_transition = self.check_transition_activation(prev_state, action, next_state)
            
            self.learn_from_experience(prev_state, action, next_state, activated_transition)
            
            prev_state = next_state
            
            if step % log_interval == 0 and step > 0:
                self._log_progress(step, total_steps)
        
        self._print_training_stats()
    
    def _log_progress(self, step, total_steps):
        print(f"Step {step}/{total_steps}")
        
        q_sums = []
        for goal_index in range(len(self.all_transitions)):
            q_sum = self.get_q_value_sum(goal_index)
            q_sums.append((goal_index, q_sum))
        
        q_sums.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 learned policies:")
        for i, (goal_index, q_sum) in enumerate(q_sums[:3]):
            goal_desc = self.get_goal_description(goal_index)
            activations = self.training_stats['transitions_activated'][goal_index]
            print(f"  {i+1}. {goal_desc[:50]}... - Q-sum: {q_sum:.1f}")
    
    def _print_training_stats(self):
        print("\nTraining Summary:")
        print(f"Steps: {self.training_stats['steps']}")
        print(f"Valid moves: {self.training_stats['valid_moves']}")
        print(f"Q-updates: {self.training_stats['q_updates']}")
        
        active_transitions = [(i, count) for i, count in enumerate(self.training_stats['transitions_activated']) if count > 0]
        print(f"Activated {len(active_transitions)}/{len(self.all_transitions)} transitions")
    
    def get_policy(self, goal_index, state):
        state_key = self.get_state_key(state)
        q_values = self.q_tables[goal_index][state_key]
        
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return 4
        
        best_action = None
        best_q = -float('inf')
        
        for action in valid_actions:
            if q_values[action] > best_q:
                best_q = q_values[action]
                best_action = action
        
        if best_q <= 0:
            best_action = random.choice(valid_actions)
        
        if self.debug:
            print(f"  Policy for goal {goal_index}: state {state} -> action {best_action}")
        
        return best_action
    
    def get_goal_description(self, goal_index):
        transition = self.all_transitions[goal_index]
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "TOGGLE"}
        action_name = action_names.get(transition['action'], str(transition['action']))
        return f"{transition['type']}: {transition['prev_position']} --{action_name}--> {transition['next_position']}"
    
    def test_single_policy(self, goal_index, start_position, max_steps=100):
        print(f"Testing policy {goal_index} from {start_position}")
        
        original_pos = self.grid_world.agent_pos
        original_inventory = self.grid_world.agent_inventory
        
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
            
            activated = self.check_transition_activation(prev_state, action, state)
            if activated == goal_index:
                print(f"SUCCESS: Goal achieved in {step+1} steps")
                break
            elif activated is not None:
                print(f"WRONG: Activated {activated} instead of {goal_index}")
                break
        
        self.grid_world.agent_pos = original_pos
        self.grid_world.agent_inventory = original_inventory
        
        return path

    def get_q_value_sum(self, goal_index):
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
            print(f"Created {len(self.operators)} operators")
    
    def _create_operators(self):
        operators = []
        
        transitions = self.grid_world.get_important_transitions()
        door_transitions = transitions['door_transitions']
        object_transitions = transitions['object_transitions']
        
        # FIXED: Correct room assignment for door transitions
        for transition in door_transitions:
            # Get the actual rooms from the positions
            from_pos = transition['prev_position']
            to_pos = transition['next_position']
            
            from_room = self.grid_world.room_ids.get(from_pos)
            to_room = self.grid_world.room_ids.get(to_pos)
            
            # Only create operator if both rooms are valid
            if from_room is not None and to_room is not None:
                preconditions = [
                    f"agent_in_room({from_room})",
                    f"connected({from_room}, {to_room})"
                ]
                
                operators.append(("MOVE", preconditions, transition))
        
        # FIXED: Correct object transitions
        for transition in object_transitions:
            pos = transition['prev_position']
            room_id = self.grid_world.room_ids.get(pos)
            
            if room_id is not None:
                # PICK_UP operator - agent must be in room, object present, empty inventory
                pick_up_preconditions = [
                    f"agent_in_room({room_id})",
                    f"object_at_position({pos[0]}, {pos[1]})",
                    f"inventory_empty()"
                ]
                
                operators.append(("PICK_UP", pick_up_preconditions, transition))
                
                # PUT_DOWN operator - agent must be in room, holding object, position empty
                put_down_preconditions = [
                    f"agent_in_room({room_id})",
                    f"position_empty({pos[0]}, {pos[1]})",
                    f"inventory_has_object()"
                ]
                
                operators.append(("PUT_DOWN", put_down_preconditions, transition))
        
        return operators
    
    def get_operators_by_action(self, action_name):
        return [op for op in self.operators if op[0] == action_name]
    
    def get_operator_by_transition(self, transition):
        for op in self.operators:
            if op[2] == transition:
                return op
        return None
    
    def find_applicable_operators(self, state):
        applicable = []
        
        for operator in self.operators:
            action_name, preconditions, transition = operator
            
            if self._check_preconditions(state, preconditions):
                applicable.append(operator)
        
        return applicable
    
    def _check_preconditions(self, state, preconditions):
        # Extract state information
        if 'agent_position' in state:
            agent_pos = state['agent_position']
            agent_room = self.grid_world.room_ids.get(agent_pos, -1)
            agent_inventory = state.get('inventory', None)
            grid = state.get('grid', self.grid_world.grid)
        else:
            # Assume it's a planning state
            agent_room = state['agent_location']
            agent_inventory = state['agent_inventory']
            grid = self.grid_world.grid
        
        # Build room connections from door transitions
        room_connections = defaultdict(set)
        for transition in self.grid_world.door_transitions:
            from_pos = transition['prev_position']
            to_pos = transition['next_position']
            from_room = self.grid_world.room_ids.get(from_pos)
            to_room = self.grid_world.room_ids.get(to_pos)
            if from_room is not None and to_room is not None:
                room_connections[from_room].add(to_room)
                room_connections[to_room].add(from_room)
        
        for precondition in preconditions:
            if precondition.startswith("agent_in_room("):
                room_num = int(precondition.split('(')[1].split(')')[0])
                if agent_room != room_num:
                    return False
                    
            elif precondition.startswith("connected("):
                rooms_str = precondition.split('(')[1].split(')')[0]
                parts = rooms_str.split(',')
                room1 = int(parts[0].strip())
                room2 = int(parts[1].strip())
                if room2 not in room_connections.get(room1, set()):
                    return False
                    
            elif precondition.startswith("object_at_position("):
                # Extract coordinates from precondition like "object_at_position(4, 1)"
                coords_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, coords_str.split(','))
                if grid[x, y] == 0:  # No object at position
                    return False
                    
            elif precondition.startswith("position_empty("):
                # Extract coordinates from precondition like "position_empty(4, 1)"
                coords_str = precondition.split('(')[1].split(')')[0]
                x, y = map(int, coords_str.split(','))
                if grid[x, y] != 0:  # Position not empty
                    return False
                    
            elif precondition.startswith("inventory_empty()"):
                if agent_inventory is not None:
                    return False
                    
            elif precondition.startswith("inventory_has_object()"):
                if agent_inventory is None:
                    return False
        
        return True
    
    def display_operators(self):
        print("Action Operators:")
        
        actions = {}
        for op in self.operators:
            action_name = op[0]
            if action_name not in actions:
                actions[action_name] = []
            actions[action_name].append(op)
        
        for action_name, operators in actions.items():
            print(f"  {action_name}: {len(operators)} operators")
            for op in operators:  # Show first 3 of each type
                _, preconditions, transition = op
                print(f"    - {preconditions} -> {transition['type']} at {transition['prev_position']}")
    
    def get_state_description(self):
        agent_room = self.grid_world.room_ids[self.grid_world.agent_pos]
        
        object_locations = {}
        for table_pos in self.grid_world.table_positions.keys():
            x, y = table_pos
            if self.grid_world.grid[x, y] != 0:
                obj_id = self.grid_world.grid[x, y]
                room_id = self.grid_world.room_ids[table_pos]
                object_locations[obj_id] = room_id
        
        room_connections = defaultdict(set)
        for transition in self.grid_world.door_transitions:
            from_pos = transition['prev_position']
            to_pos = transition['next_position']
            from_room = self.grid_world.room_ids.get(from_pos)
            to_room = self.grid_world.room_ids.get(to_pos)
            if from_room is not None and to_room is not None:
                room_connections[from_room].add(to_room)
                room_connections[to_room].add(from_room)
        
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
        
        return planning_state
    
    def test_preconditions(self, position):
        original_pos = self.grid_world.agent_pos
        self.grid_world.agent_pos = position
        
        state = {
            'agent_position': position,
            'inventory': self.grid_world.agent_inventory,
            'grid': self.grid_world.grid
        }
        applicable = self.find_applicable_operators(state)
        
        print(f"From {position} (room {self.grid_world.room_ids[position]}): {len(applicable)} applicable operators")
        for op in applicable:
            action_name, preconditions, transition = op
            print(f"  - {action_name}: {preconditions}")
        
        self.grid_world.agent_pos = original_pos
        
        return applicable
    

class IntegratedPlanner:
    def __init__(self, grid_world, action_operators, q_learning_agent, debug=False):
        self.grid_world = grid_world
        self.action_ops = action_operators
        self.q_agent = q_learning_agent
        self.domain = PlanningDomain()
        self.debug = debug
        
        self.transition_to_operator = {}
        for i, op in enumerate(self.action_ops.operators):
            transition_key = self._get_transition_key(op[2])
            self.transition_to_operator[transition_key] = i
        
        if self.debug:
            print(f"IntegratedPlanner with {len(self.action_ops.operators)} operators")
    
    def _get_transition_key(self, transition):
        return (transition['prev_position'], transition['action'], transition['next_position'])
    
    def _get_transition_index(self, transition):
        for i, t in enumerate(self.q_agent.all_transitions):
            if (t['prev_position'] == transition['prev_position'] and 
                t['action'] == transition['action'] and
                t['next_position'] == transition['next_position']):
                return i
        return None
    
    def is_operator_achievable(self, operator, current_state):
        _, _, transition = operator
        transition_index = self._get_transition_index(transition)
        
        if transition_index is None:
            return False
        
        current_pos = current_state['agent_position']
        max_q = np.max(self.q_agent.q_tables[transition_index][current_pos])
        
        return max_q > 0
    
    def get_achievable_operators(self, current_state):
        applicable = self.action_ops.find_applicable_operators(current_state)
        achievable = []
        
        for op in applicable:
            if self.is_operator_achievable(op, current_state):
                achievable.append(op)
        
        return achievable
    
    def create_planning_state(self, current_state, activated_transition=None):
        if 'agent_position' in current_state:
            if activated_transition is not None:
                new_position = activated_transition['next_position']
            else:
                new_position = current_state['agent_position']
            agent_inventory = current_state.get('inventory', None)
        else:
            if activated_transition is not None:
                new_position = activated_transition['next_position']
            else:
                new_position = current_state['low_level_position']
            agent_inventory = current_state['agent_inventory']
        
        room_id = self.grid_world.room_ids[new_position]
        
        room_connections = defaultdict(set)
        for transition in self.q_agent.all_transitions:
            if transition['type'] == 'door':
                from_room = self.grid_world.room_ids[transition['prev_position']]
                to_room = self.grid_world.room_ids[transition['next_position']]
                room_connections[from_room].add(to_room)
                room_connections[to_room].add(from_room)
        
        tables = set()
        for table_pos in self.grid_world.table_positions.keys():
            room_id_table = self.grid_world.room_ids[table_pos]
            tables.add(room_id_table)
        
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
        
        return planning_state
    
    def plan_with_learned_policies(self, goal, max_depth=20):
        current_state = self.action_ops.get_state_description()
        planning_state = self.create_planning_state(current_state)
        
        print("Planning from:")
        print(f"  Room: {planning_state['agent_location']}")
        print(f"  Inventory: {planning_state['agent_inventory']}")
        
        if self.domain.is_goal_state(planning_state, goal):
            print("Goal already satisfied!")
            return []
        
        queue = deque([(planning_state, [], planning_state['low_level_position'])])
        visited = set()
        
        states_explored = 0
        
        while queue:
            state, plan, low_level_pos = queue.popleft()
            states_explored += 1
            
            if self.domain.is_goal_state(state, goal):
                print(f"Found plan with {len(plan)} steps (explored {states_explored} states)")
                return plan
            
            if len(plan) >= max_depth:
                continue
            
            state_key = (
                state['agent_location'],
                state['agent_inventory'],
                tuple(sorted(state['object_locations'].items()))
            )
            if state_key in visited:
                continue
            visited.add(state_key)
            
            current_low_level_state = {
                'agent_position': low_level_pos,
                'inventory': state['agent_inventory'],
                'grid': self.grid_world.grid.copy()
            }
            
            achievable_ops = self.get_achievable_operators(current_low_level_state)
            
            for op in achievable_ops:
                action_name, preconditions, transition = op
                transition_index = self._get_transition_index(transition)
                
                new_state = None
                if action_name == "MOVE":
                    from_room = self.grid_world.room_ids[transition['prev_position']]
                    to_room = self.grid_world.room_ids[transition['next_position']]
                    
                    new_state = copy.deepcopy(state)
                    new_state['agent_location'] = to_room
                    new_low_level_pos = transition['next_position']
                    
                    action_desc = f"MOVE from room {from_room} to room {to_room}"
                    params = {'from_room': from_room, 'to_room': to_room}
                    
                elif action_name == "PICK_UP":
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
                    new_low_level_pos = transition['next_position']
                    
                    action_desc = f"PICK_UP object {obj_id} in room {room_id}"
                    params = {'object_id': obj_id, 'room': room_id}
                    
                elif action_name == "PUT_DOWN":
                    if state['agent_inventory'] is None:
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
                
                new_plan = plan + [(action_name, params, action_desc, transition_index)]
                queue.append((new_state, new_plan, new_low_level_pos))
        
        print(f"No plan found after {states_explored} states")
        return None
    
    def execute_integrated_plan(self, plan, max_steps_per_action=100):
        if not plan:
            return True, "No plan needed"
        
        print(f"Executing {len(plan)}-step plan")
        
        for i, (action_name, params, description, transition_index) in enumerate(plan):
            print(f"Step {i+1}: {description}")
            
            success = self._execute_transition_policy(transition_index, max_steps_per_action)
            
            if not success:
                return False, f"Failed to execute {description}"
            
            if not self._verify_operator_applied(action_name, params):
                return False, f"Operator {action_name} did not produce expected result"
        
        return True, "Plan executed successfully"
    
    def _execute_transition_policy(self, transition_index, max_steps):
        transition = self.q_agent.all_transitions[transition_index]
        target_position = transition['prev_position']
        
        steps = 0
        while self.grid_world.agent_pos != target_position and steps < max_steps:
            current_pos = self.grid_world.agent_pos
            action = self.q_agent.get_policy(transition_index, current_pos)
            
            self.grid_world.step(action)
            steps += 1
        
        if self.grid_world.agent_pos != target_position:
            return False
        
        required_action = transition['action']
        self.grid_world.step(required_action)
        
        return self.grid_world.agent_pos == transition['next_position']
    
    def _verify_operator_applied(self, action_name, params):
        if action_name == "MOVE":
            expected_room = params['to_room']
            current_room = self.grid_world.room_ids[self.grid_world.agent_pos]
            return current_room == expected_room
            
        elif action_name == "PICK_UP":
            expected_obj = params['object_id']
            return self.grid_world.agent_inventory == expected_obj
            
        elif action_name == "PUT_DOWN":
            expected_obj = params['object_id']
            expected_room = params['room']
            
            for table_pos, room_coords in self.grid_world.table_positions.items():
                if self.grid_world.room_ids[table_pos] == expected_room:
                    if self.grid_world.grid[table_pos] == expected_obj:
                        return True
            return False
        
        return True
    
    def test_planning_from_position(self, goal, position):
        print(f"Testing planning from {position}")
        
        original_pos = self.grid_world.agent_pos
        self.grid_world.agent_pos = position
        
        current_state = {'agent_position': position, 'inventory': self.grid_world.agent_inventory}
        achievable = self.get_achievable_operators(current_state)
        print(f"Achievable operators: {len(achievable)}")
        
        plan = self.plan_with_learned_policies(goal, max_depth=30)
        
        if plan:
            print(f"Found {len(plan)}-step plan")
            for i, (action_name, params, description, _) in enumerate(plan):
                print(f"  {i+1}. {description}")
        else:
            print("No plan found")
        
        self.grid_world.agent_pos = original_pos
        
        return plan

def demonstrate_integrated_planner():
    print("=== INTEGRATED PLANNER DEMONSTRATION ===\n")
    
    # Step 1: Create the grid world
    print("1. Creating Grid World (4 rooms, room_size=3)...")
    grid_world = GridWorld(num_rooms=4, room_size=3, debug=False)
    
    # Display initial state
    print("Initial grid state:")
    grid_world.render()
    print(grid_world.door_transitions)
    print()
    
    # Step 2: Create action operators
    print("2. Creating Action Operators...")
    action_ops = ActionOperators(grid_world, debug=True)
    action_ops.display_operators()
    #print(action_ops.operators)
    
    print()
    
    # Step 3: Create and train Q-learning agent
    print("3. Training Goal-Conditioned Q-Learning Agent...")
    q_agent = GoalConditionedQLearning(grid_world, debug=False)
    
    print(q_agent.all_transitions)
    # Train for a reasonable number of steps (reduced for demo)
    q_agent.train_continuous(total_steps=5000, log_interval=1000)
    print()
    
    # Step 4: Create integrated planner
    print("4. Creating Integrated Planner...")
    integrated_planner = IntegratedPlanner(grid_world, action_ops, q_agent, debug=True)
    
    # Step 5: Test current state and achievable operators
    print("5. Testing Current State Analysis...")
    current_state = action_ops.get_state_description()
    print(f"Current agent state:")
    print(f"  Position: {grid_world.agent_pos}")
    print(f"  Room: {current_state['agent_location']}")
    print(f"  Inventory: {current_state['agent_inventory']}")
    print(f"  Known object locations: {current_state['object_locations']}")
    
    # Test achievable operators from current position
    test_state = {'agent_position': grid_world.agent_pos, 'inventory': grid_world.agent_inventory}
    achievable_ops = integrated_planner.get_achievable_operators(test_state)
    print(f"Achievable operators from current position: {len(achievable_ops)}")
    for op in achievable_ops[:3]:  # Show first 3
        print(f"  - {op[0]} at {op[2]['prev_position']}")
    print()
    
    # Step 6: Define test goals and plan
    print("6. Testing Planning with Different Goals...")
    
    # Goal 1: Move to a specific room
    print("\n--- Goal 1: Move to Room 2 ---")
    goal1 = {'object_location': {'object_id': 1, 'room': 2}}  # Using object location as proxy for room goal
    plan1 = integrated_planner.test_planning_from_position(goal1, grid_world.agent_pos)
    
    if plan1:
        print("Executing plan to reach room 2...")
        success, message = integrated_planner.execute_integrated_plan(plan1, max_steps_per_action=50)
        print(f"Result: {success} - {message}")
        print(f"Final position: {grid_world.agent_pos}")
        print(f"Final room: {grid_world.room_ids[grid_world.agent_pos]}")
    else:
        print("No plan found for Goal 1")
    print()
    
    # Save state after first goal
    intermediate_pos = grid_world.agent_pos
    intermediate_inventory = grid_world.agent_inventory
    
    # Goal 2: Pick up an object (if we're in a room with one)
    print("--- Goal 2: Pick up an object ---")
    
    # Find an object in the current room
    current_room = grid_world.room_ids[grid_world.agent_pos]
    object_in_room = None
    for table_pos, room_coords in grid_world.table_positions.items():
        room_id = grid_world.room_ids[table_pos]
        if room_id == current_room and grid_world.grid[table_pos] != 0:
            object_in_room = grid_world.grid[table_pos]
            break
    
    if object_in_room:
        print(f"Found object {object_in_room} in current room {current_room}")
        
        # For pickup goal, we need to be in the room and have empty inventory
        grid_world.agent_inventory = None  # Ensure empty inventory
        
        # The goal for pickup is that the object is in our inventory
        # We'll simulate this by checking if agent has the object
        goal2_state = action_ops.get_state_description()
        goal2_state['agent_inventory'] = object_in_room
        
        # Create a custom goal checker for pickup
        def is_pickup_goal(state, target_obj):
            return state['agent_inventory'] == target_obj
        
        print("Planning to pick up object...")
        # We'll manually test achievable operators for pickup
        test_state2 = {'agent_position': grid_world.agent_pos, 'inventory': None}
        achievable_ops2 = integrated_planner.get_achievable_operators(test_state2)
        
        pickup_ops = [op for op in achievable_ops2 if op[0] == 'PICK_UP']
        print(f"Found {len(pickup_ops)} achievable PICK_UP operations")
        
        if pickup_ops:
            # Execute the first pickup operation
            pickup_op = pickup_ops[0]
            transition_index = integrated_planner._get_transition_index(pickup_op[2])
            print(f"Executing pickup operation...")
            success = integrated_planner._execute_transition_policy(transition_index, 50)
            
            if success:
                print(f"Successfully picked up object! Inventory: {grid_world.agent_inventory}")
            else:
                print("Failed to pick up object")
    else:
        print("No objects in current room to pick up")
    print()
    
    # Goal 3: Test movement between rooms using learned policies
    print("--- Goal 3: Test Individual Transition Policies ---")
    
    # Test a specific door transition
    door_transitions = [t for t in q_agent.all_transitions if t['type'] == 'door']
    if door_transitions:
        test_transition = door_transitions[0]
        transition_index = q_agent.all_transitions.index(test_transition)
        
        print(f"Testing door transition policy {transition_index}:")
        print(f"  From: {test_transition['prev_position']}")
        print(f"  Action: {test_transition['action']}")
        print(f"  To: {test_transition['next_position']}")
        
        # Move agent to start position for testing
        grid_world.agent_pos = test_transition['prev_position']
        grid_world.agent_inventory = None
        
        path = q_agent.test_single_policy(
            transition_index, 
            test_transition['prev_position'], 
            max_steps=50
        )
        
        print(f"Path length: {len(path)}")
        print(f"Final position: {grid_world.agent_pos}")
        print(f"Target position: {test_transition['next_position']}")
        success = grid_world.agent_pos == test_transition['next_position']
        print(f"Policy test: {'SUCCESS' if success else 'FAILED'}")
    print()
    
    # Step 7: Test knowledge integration
    print("7. Testing Knowledge Integration...")
    
    # Create an agent with knowledge base
    print("Creating intelligent agent...")
    agent = Agent(grid_world)
    
    # Move around to gather knowledge
    print("Exploring to gather knowledge...")
    for action in [1, 3, 1, 3, 0, 2]:  # Simple exploration pattern
        agent.step(action)
    
    # Display gathered knowledge
    agent.render_knowledge()
    print()
    
    # Step 8: Comprehensive planning test
    print("8. Comprehensive Planning Test...")
    
    # Reset to known position
    grid_world.agent_pos = (1, 1)
    grid_world.agent_inventory = None
    
    # Create a complex goal (move object between rooms)
    print("Testing complex object relocation planning...")
    
    # Find objects and rooms for planning
    object_locations = {}
    for table_pos in grid_world.table_positions.keys():
        x, y = table_pos
        if grid_world.grid[x, y] != 0:
            obj_id = grid_world.grid[x, y]
            room_id = grid_world.room_ids[table_pos]
            object_locations[obj_id] = room_id
    
    if len(object_locations) >= 1:
        obj_id, current_room = list(object_locations.items())[0]
        target_room = (current_room + 1) % 4  # Move to next room
        
        print(f"Planning to move object {obj_id} from room {current_room} to room {target_room}")
        
        # This would require: move to room with object, pick up, move to target room, put down
        goal_complex = {'object_location': {'object_id': obj_id, 'room': target_room}}
        
        plan_complex = integrated_planner.plan_with_learned_policies(goal_complex, max_depth=30)
        
        if plan_complex:
            print(f"Found complex plan with {len(plan_complex)} steps:")
            for i, (action_name, params, description, _) in enumerate(plan_complex):
                print(f"  {i+1}. {description}")
            
            # For demonstration, we'll show the plan but not execute the full sequence
            print("(Skipping full execution for demonstration)")
        else:
            print("No plan found for complex object relocation")
    else:
        print("No objects found for complex planning test")
    
    print("\n=== DEMONSTRATION COMPLETE ===")

# Run the demonstration
if __name__ == "__main__":
    demonstrate_integrated_planner()