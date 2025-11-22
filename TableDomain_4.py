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
    A = 2
    B = 3
    C = 3
    D = 4
    E = 5
    F = 6

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
                
    
    def _precompute_transitions(self):
        """Precompute all possible door and object transitions"""        
        # Horizontal doors (vertical walls)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for a in range(4):
                    prev_pos = (i,j)
                    next_pos = self.step_2(a, i,j)

                    if (self.room_ids.get(prev_pos, -1) != self.room_ids.get(next_pos, -1) and self.grid[prev_pos] != 1 and self.grid[next_pos] != 1):
                        self.door_transitions.append({
                            'prev_position': prev_pos,
                            'action': a, # MOVE action
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

        # Update room tracking
        self.knowledge_base['previous_room'] = current_room
    
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
        
        # Put down action (can only put down if no other object in the room)
        if kb_state['inventory'] is not None:
            # Check if there are any objects already in the current room
            objects_in_room = any(room == current_room for room, obj_type in kb_state['object_locations'])
            if not objects_in_room:
                applicable.append((ActionType.PUT_DOWN, {
                    'object_type': kb_state['inventory'],
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
            state['inventory'],              # Fixed key name
            tuple(sorted(state['object_locations'])),  # Fixed: it's a set, not dict
            tuple(sorted(state['known_rooms'])),       # Added missing component
            tuple(sorted(state['room_connections']))   # Added missing component
        )

class GoalConditionedQLearning:
    """Goal-conditioned Q-learning with absorbing states for wrong transitions"""
    def __init__(self, grid_world, learning_rate=0.1, discount_factor=0.9):
        self.grid_world = grid_world
        self.alpha = learning_rate
        self.gamma = discount_factor
        
        # Get all important transitions
        transitions = grid_world.get_important_transitions()
        self.all_transitions = transitions['door_transitions'] + transitions['object_transitions']
        
        # Create Q-tables for each transition goal
        self.q_tables = {}
        for i in range(len(self.all_transitions)):
            self.q_tables[i] = defaultdict(lambda: np.zeros(5))  # 5 actions
        
        # Simple logging
        self.transitions_activated = [0] * len(self.all_transitions)
        self.steps = 0
    
    def check_transition_activation(self, prev_state, action, next_state):
        """Check if any important transition was activated"""
        for i, transition in enumerate(self.all_transitions):
            if (transition['prev_position'] == prev_state and 
                transition['action'] == action and
                transition['next_position'] == next_state):
                self.transitions_activated[i] += 1
                return i
        return None
    
    def learn_from_experience(self, prev_state, action, next_state, activated_transition):
        """Update ALL goal policies with absorbing states for wrong transitions"""
        for goal_index in range(len(self.all_transitions)):
            # Reward is 1 only if this is the goal transition
            reward = 1.0 if activated_transition == goal_index else 0.0
            
            # Get current Q-value
            current_q = self.q_tables[goal_index][prev_state][action]
            
            # Calculate target Q-value
            if activated_transition is not None:
                target = reward
            else:
                # Normal case: bootstrap from next state
                next_max = np.max(self.q_tables[goal_index][next_state])
                target = reward + self.gamma * next_max
            
            # Update Q-value
            new_q = current_q + self.alpha * (target - current_q)
            self.q_tables[goal_index][prev_state][action] = new_q
    
    def train(self, total_steps=100000):
        """Simple training with random exploration"""
        print(f"Training {len(self.all_transitions)} policies for {total_steps} steps...")
        
        prev_state = self.grid_world.agent_pos
        
        for step in range(total_steps):
            self.steps += 1
            
            # Always choose random action (0-4)
            action = random.randint(0, 4)
            
            # Take action
            self.grid_world.step(action)
            next_state = self.grid_world.agent_pos
            
            # Check transition activation
            activated_transition = self.check_transition_activation(prev_state, action, next_state)
            
            # Learn from experience
            self.learn_from_experience(prev_state, action, next_state, activated_transition)
            
            prev_state = next_state
            
            # Minimal logging
            if step % 20000 == 0:
                activated = sum(1 for count in self.transitions_activated if count > 0)
                print(f"Step {step}: Activated {activated}/{len(self.all_transitions)} transitions")
        
        # Final summary
        activated = sum(1 for count in self.transitions_activated if count > 0)
        print(f"Final: Activated {activated}/{len(self.all_transitions)} transitions")
    
    def get_policy(self, goal_index, position):
        """Get best action for a given goal and position"""
        q_values = self.q_tables[goal_index][position]
        return np.argmax(q_values)
    
    def test_policy(self, goal_index, start_position, max_steps=50):
        """Test a single policy"""
        original_pos = self.grid_world.agent_pos
        
        self.grid_world.agent_pos = start_position
        state = start_position
        
        for step in range(max_steps):
            action = self.get_policy(goal_index, state)
            prev_state = state
            self.grid_world.step(action)
            state = self.grid_world.agent_pos
            
            activated = self.check_transition_activation(prev_state, action, state)
            if activated == goal_index:
                print(f"Success in {step+1} steps!")
                break
        
        self.grid_world.agent_pos = original_pos

class LearningAgent(Agent):
    """Agent subclass that integrates GoalConditionedQLearning while exploring randomly"""
    
    def __init__(self, grid_world, learning_rate=0.1, discount_factor=0.9):
        super().__init__(grid_world)
        
        # Initialize Q-learning
        self.q_learner = GoalConditionedQLearning(
            grid_world, 
            learning_rate=learning_rate, 
            discount_factor=discount_factor
        )
        
        # Track which transitions we've encountered
        self.encountered_transitions = set()
        self.total_steps = 0
        
        # Simple count-based exploration: track state-action counts
        self.state_action_counts = np.zeros((grid_world.grid_size, grid_world.grid_size, 5), dtype=int)
        
        print(f"LearningAgent initialized with {len(self.q_learner.all_transitions)} transitions to learn")
    
    def step(self, action):
        """Override step to include Q-learning updates"""
        prev_state = self.grid_world.agent_pos
        prev_room = self.grid_world.get_current_room_id()
        
        # Update count before taking action
        self.state_action_counts[prev_state[0], prev_state[1], action] += 1
        
        # Execute the action using parent class
        super().step(action)
        
        next_state = self.grid_world.agent_pos
        next_room = self.grid_world.get_current_room_id()
        
        # Check if this action activated any important transition
        activated_transition = self.q_learner.check_transition_activation(
            prev_state, action, next_state
        )
        
        # Track encountered transitions
        if activated_transition is not None:
            self.encountered_transitions.add((prev_state, action, next_state))
        
        # Learn from this experience (update all goal policies)
        self.q_learner.learn_from_experience(
            prev_state, action, next_state, activated_transition
        )
        
        self.total_steps += 1
    
    def choose_action_count_based(self):
        """Simple count-based exploration: choose the least taken action in current state"""
        
        x, y = self.grid_world.agent_pos
        
        return np.argmin(self.state_action_counts[x, y, :])
        
    
    def explore_count_based(self, num_steps=10000, log_interval=1000):
        """Explore using simple count-based exploration"""
        print(f"Starting count-based exploration for {num_steps} steps...")
        
        for step in range(num_steps):
            action = self.choose_action_count_based()
            self.step(action)
            
            if step % log_interval == 0:
                self._log_progress(step)
        
        self._log_progress(num_steps)
        print("Count-based exploration completed!")
    
    def _log_progress(self, step):
        """Log current learning and knowledge progress"""
        activated = len(self.encountered_transitions)
        total_transitions = len(self.q_learner.all_transitions)
        #print(self.encountered_transitions)
        #print(self.q_learner.all_transitions)
        known_rooms = len(self.knowledge_base['known_rooms'])
        known_objects = len(self.knowledge_base['object_locations'])
        known_connections = len(self.knowledge_base['room_connections'])
        
        print(f"Step {step}: Known {known_rooms} rooms, {known_connections} connections, "
              f"{known_objects} objects, learned {activated}/{total_transitions} transitions")
    
    
    def get_learning_progress(self):
        """Get current learning progress statistics"""
        return {
            'total_steps': self.total_steps,
            'encountered_transitions': len(self.encountered_transitions),
            'total_transitions': len(self.q_learner.all_transitions),
            'known_rooms': len(self.knowledge_base['known_rooms']),
            'known_objects': len(self.knowledge_base['object_locations']),
            'known_connections': len(self.knowledge_base['room_connections'])
        }
    
# Create environment and agent
grid_world = GridWorld(num_rooms=64, room_size=5, debug=False)
agent = LearningAgent(grid_world)

# Use simple count-based exploration
agent.explore_count_based(num_steps=500000, log_interval=20000)
