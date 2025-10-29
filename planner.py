import random
import math
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple, Any, Dict, DefaultDict
from collections import deque, defaultdict
import time

@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Tuple]  # First-order logic predicates

class SpatialDomain:
    VALID_ACTIONS = {'goto', 'pickup', 'putdown', 'placenear'}
    
    def __init__(self, locations: Set[str], objects: Set[str]):
        self.locations = locations
        self.objects = objects

    # ====== Predicate Evaluation Methods ======
    def get_agent_location(self, state: State) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == 'agent_at' and len(pred) == 2:
                return pred[1]
        return None

    def is_holding(self, state: State) -> bool:
        return any(p[0] == 'holding' for p in state.predicates)
    
    def get_held_object(self, state: State) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == 'holding' and len(pred) == 2:
                return pred[1]
        return None

    def get_object_location(self, state: State, obj: str) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == 'at' and pred[1] == obj and len(pred) == 3:
                return pred[2]
        return None

    def are_near(self, state: State, obj1: str, obj2: str) -> bool:
        return (('near', obj1, obj2) in state.predicates) or \
               (('near', obj2, obj1) in state.predicates)

    # ====== Action Implementation ======
    def goto(self, state: State, new_loc: str) -> Optional[State]:
        if new_loc not in self.locations:
            return None
            
        current_loc = self.get_agent_location(state)
        if current_loc == new_loc:
            return None
            
        new_predicates = set(state.predicates)
        if current_loc:
            new_predicates.discard(('agent_at', current_loc))
        new_predicates.add(('agent_at', new_loc))
        return State(frozenset(new_predicates))

    def pickup(self, state: State, obj: str) -> Optional[State]:
        if self.is_holding(state):
            return None
            
        agent_loc = self.get_agent_location(state)
        obj_loc = self.get_object_location(state, obj)
        
        if agent_loc is None or obj_loc != agent_loc:
            return None
            
        new_predicates = set(state.predicates)
        new_predicates.discard(('at', obj, obj_loc))
        # Remove all near predicates that involve the object being picked up
        for pred in list(new_predicates):  # Use list to avoid modification during iteration
            if pred[0] == 'near' and (pred[1] == obj or pred[2] == obj):
                new_predicates.discard(pred)
        new_predicates.add(('holding', obj))
        return State(frozenset(new_predicates))

    def putdown(self, state: State) -> Optional[State]:
        obj = self.get_held_object(state)
        if obj is None:
            return None
            
        agent_loc = self.get_agent_location(state)
        if agent_loc is None:
            return None
            
        new_predicates = set(state.predicates)
        new_predicates.discard(('holding', obj))
        new_predicates.add(('at', obj, agent_loc))
        return State(frozenset(new_predicates))

    def placenear(self, state: State, target_obj: str) -> Optional[State]:
        held_obj = self.get_held_object(state)
        if held_obj is None:
            return None
            
        target_loc = self.get_object_location(state, target_obj)
        if target_loc is None:
            return None
            
        agent_loc = self.get_agent_location(state)
        if agent_loc != target_loc:
            return None
            
        new_predicates = set(state.predicates)
        new_predicates.discard(('holding', held_obj))
        new_predicates.add(('at', held_obj, agent_loc))
        new_predicates.add(('near', held_obj, target_obj))
        return State(frozenset(new_predicates))

    # ====== Logical Formula Evaluation ======
    def evaluate_formula(self, state: State, formula: Any) -> bool:
        if isinstance(formula, tuple):
            operator = formula[0]
            
            if operator == 'not':
                return not self.evaluate_formula(state, formula[1])
                
            elif operator == 'and':
                return all(self.evaluate_formula(state, f) for f in formula[1:])
                
            elif operator == 'or':
                return any(self.evaluate_formula(state, f) for f in formula[1:])
                
            elif operator == 'agent_at':
                return self.get_agent_location(state) == formula[1]
                
            elif operator == 'holding':
                if len(formula) == 1:
                    return self.is_holding(state)
                return self.get_held_object(state) == formula[1]
                
            elif operator == 'at':
                return self.get_object_location(state, formula[1]) == formula[2]
                
            elif operator == 'near':
                return self.are_near(state, formula[1], formula[2])
                
            return formula in state.predicates
            
        return formula in state.predicates

    # ====== Planning Algorithm ======
    def find_plan(
        self,
        initial_state: State,
        goal_formula: Any,
        max_depth: int = 100,  # Reduced from 1000 to prevent infinite loops,
        print_planning: bool = False
    ) -> List[str]:
        queue = deque([(initial_state, [])])
        visited = set()
        visited.add(initial_state)
        i = 0
        while queue:
            state, path = queue.popleft()

            if print_planning:
                i+=1
                print("-----------------------")
                print(f"Planning state number: {i}")
                print(state, path)
                print("++++++++++++++++++++++++++")

            if self.evaluate_formula(state, goal_formula):
                return path
                
            if len(path) >= max_depth:
                continue
                
            # Generate actions in a priority order to find solutions faster
            actions = self._generate_actions(state)
            
            # Prioritize actions that directly contribute to the goal
            prioritized_actions = []
            other_actions = []
            
            for action in actions:
                if action.startswith("pickup") or action.startswith("placenear"):
                    # Check if this action directly contributes to the goal
                    parts = action.split()
                    if len(parts) >= 2:
                        obj = parts[1]
                        # Simple heuristic: prioritize actions involving objects mentioned in the goal
                        if self._is_object_in_goal(obj, goal_formula):
                            prioritized_actions.append(action)
                            continue
                other_actions.append(action)
            
            # Process prioritized actions first
            for action in prioritized_actions + other_actions:
                new_state = self.execute(state, action)
                #print(new_state, state)
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action]))
        
        return []

    def _is_object_in_goal(self, obj: str, goal_formula: Any) -> bool:
        """Check if an object is mentioned in the goal formula"""
        if isinstance(goal_formula, tuple):
            # Recursively check all parts of the formula
            for part in goal_formula:
                if self._is_object_in_goal(obj, part):
                    return True
        elif isinstance(goal_formula, str):
            return obj in goal_formula
        return False

    def _generate_actions(self, state: State) -> List[str]:
        actions = []
        agent_loc = self.get_agent_location(state)
        
        # Movement actions - only generate moves to locations with objects
        object_locations = set()
        for obj in self.objects:
            obj_loc = self.get_object_location(state, obj)
            if obj_loc:
                object_locations.add(obj_loc)
        
        for loc in self.locations:
            if loc != agent_loc and loc in object_locations:
                actions.append(f"goto {loc}")
        
        # Pickup actions
        if not self.is_holding(state):
            for obj in self.objects:
                if self.get_object_location(state, obj) == agent_loc:
                    actions.append(f"pickup {obj}")
        
        # Putdown action
        if self.is_holding(state):
            actions.append("putdown")
        
        # Placenear actions
        if self.is_holding(state):
            held_obj = self.get_held_object(state)
            for target_obj in self.objects:
                if (held_obj != target_obj and 
                    self.get_object_location(state, target_obj) == agent_loc):
                    actions.append(f"placenear {target_obj}")
                    
        return actions

    def execute(self, state: State, action: str) -> Optional[State]:
        parts = action.split()
        if not parts:
            return None
            
        cmd = parts[0]
        if cmd == "goto" and len(parts) == 2:
            return self.goto(state, parts[1])
        elif cmd == "pickup" and len(parts) == 2:
            return self.pickup(state, parts[1])
        elif cmd == "putdown" and len(parts) == 1:
            return self.putdown(state)
        elif cmd == "placenear" and len(parts) == 2:
            return self.placenear(state, parts[1])
        return None

class GridWorld:
    def __init__(self, width=20, height=20, perception_radius=3):
        self.width = width
        self.height = height
        self.perception_radius = perception_radius
        
        # Define 4 locations (A, B, C, D) as squares in the grid
        self.locations = {
            'A': (0, 0, width//2, height//2),
            'B': (width//2, 0, width, height//2),
            'C': (0, height//2, width//2, height),
            'D': (width//2, height//2, width, height)
        }
        
        # Center point of each location (using integer division)
        self.location_centers = {
            'A': (width//4, height//4),
            'B': (3*width//4, height//4),
            'C': (width//4, 3*height//4),
            'D': (3*width//4, 3*height//4)
        }
        
        # Objects in the world
        self.objects = {"book", "cup", "apple", "ball", "key"}
        self.object_positions = {}
        
        # Agent position
        self.agent_position = (width//2, height//2)  # Center of grid
        
        # Initialize object positions
        self.reset_world()
        
    def reset_world(self):
        # Place objects randomly in the four locations
        self.object_positions = {}
        for obj in self.objects:
            loc = random.choice(list(self.locations.keys()))
            x1, y1, x2, y2 = self.locations[loc]
            # Ensure objects are placed within the bounds (exclusive of upper bounds)
            x = random.randint(x1, x2 - 1)
            y = random.randint(y1, y2 - 1)
            self.object_positions[obj] = (x, y, loc)
    
    def get_agent_location(self):
        # Determine which location the agent is in
        x, y = self.agent_position
        for loc, (x1, y1, x2, y2) in self.locations.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return loc
        return None
    
    def move_agent(self, dx, dy):
        new_x = self.agent_position[0] + dx
        new_y = self.agent_position[1] + dy
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.agent_position = (new_x, new_y)
            return True
        return False
    
    def goto_location(self, location):
        # Move agent to the center of the specified location
        if location in self.location_centers:
            self.agent_position = self.location_centers[location]
            return True
        return False
    
    def get_perceived_objects(self):
        # Get objects within perception radius of agent
        perceived = {}
        agent_x, agent_y = self.agent_position
        
        for obj, (x, y, loc) in self.object_positions.items():
            distance = math.sqrt((x - agent_x)**2 + (y - agent_y)**2)
            if distance <= self.perception_radius:
                perceived[obj] = (x, y, loc)  # Store exact position and location
                
        return perceived
    
    def pickup_object(self, obj):
        # Check if object is in the same location as agent
        agent_loc = self.get_agent_location()
        if obj in self.object_positions:
            _, _, obj_loc = self.object_positions[obj]
            if obj_loc == agent_loc:
                # Remove object from world (agent picks it up)
                del self.object_positions[obj]
                return True
        return False
    
    def putdown_object(self, obj):
        # Place object at agent's current location
        agent_loc = self.get_agent_location()
        if agent_loc is not None:
            x, y = self.agent_position
            self.object_positions[obj] = (x, y, agent_loc)
            return True
        return False

class Agent:
    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.knowledge_base = {
            'objects': {},  # obj: (x, y, location) - exact position and location
            'agent_loc': grid_world.get_agent_location()
        }
        self.visitation_counts = defaultdict(int)
        self.held_object = None
        
        # Initialize visitation count for starting position
        self.visitation_counts[self.grid_world.agent_position] = 1
        
        # Create spatial domain
        self.domain = SpatialDomain(
            locations=set(grid_world.locations.keys()),
            objects=grid_world.objects
        )
        
        # Initial state for planning
        self.current_state = self._get_current_state()
    
    def _get_current_state(self):
        # Create state based on knowledge
        predicates = set()
        
        # Agent location
        predicates.add(('agent_at', self.knowledge_base['agent_loc']))
        
        # Object locations (only location, not exact position)
        for obj, (x, y, loc) in self.knowledge_base['objects'].items():
            predicates.add(('at', obj, loc))
            
        # Holding status
        if self.held_object:
            predicates.add(('holding', self.held_object))
            
        # Near relationships
        if 'near' in self.knowledge_base:
            for obj1, obj2 in self.knowledge_base['near']:
                predicates.add(('near', obj1, obj2))
                
        return State(frozenset(predicates))
    
    def update_knowledge(self):
        # Update knowledge based on perception
        perceived_objects = self.grid_world.get_perceived_objects()
        for obj, (x, y, loc) in perceived_objects.items():
            self.knowledge_base['objects'][obj] = (x, y, loc)
            
        self.knowledge_base['agent_loc'] = self.grid_world.get_agent_location()
        
        # Update current state
        self.current_state = self._get_current_state()
    
    def choose_direction(self):
        """Choose direction based on least-visited adjacent cells"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbor_counts = []
        x, y = self.grid_world.agent_position
        
        # Evaluate all possible moves
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_world.width and 0 <= ny < self.grid_world.height:
                count = self.visitation_counts.get((nx, ny), 0)
                neighbor_counts.append((count, dx, dy))
        
        # Find minimum visitation count
        if not neighbor_counts:
            return (0, 0)  # Can't move
        min_count = min([c[0] for c in neighbor_counts])
        candidates = [(dx, dy) for cnt, dx, dy in neighbor_counts if cnt == min_count]
        return random.choice(candidates)
    
    def systematic_exploration(self):
        """Move to the least visited adjacent cell"""
        dx, dy = self.choose_direction()
        if self.grid_world.move_agent(dx, dy):
            # Update visitation count
            self.visitation_counts[self.grid_world.agent_position] += 1
            return True, 1  # Return success and cost of 1 step
        return False, 0
    
    def find_plan(self, goal_formula, print_planning = False):
        return self.domain.find_plan(self.current_state, goal_formula, print_planning = print_planning)
    
    def execute_action(self, action: str):
        parts = action.split()
        if not parts:
            return False, 0
            
        cmd = parts[0]
        cost = 0
        
        if cmd == "goto" and len(parts) == 2:
            location = parts[1]
            # Calculate Manhattan distance to location center
            current_x, current_y = self.grid_world.agent_position
            target_x, target_y = self.grid_world.location_centers[location]
            distance = abs(current_x - target_x) + abs(current_y - target_y)
            
            if self.grid_world.goto_location(location):
                cost = distance
                self.update_knowledge()
                return True, cost
                
        elif cmd == "pickup" and len(parts) == 2:
            obj = parts[1]
            # Check if we know the object's exact position
            if obj not in self.knowledge_base['objects']:
                return False, 0
                
            # Get object's exact position
            obj_x, obj_y, obj_loc = self.knowledge_base['objects'][obj]
            agent_x, agent_y = self.grid_world.agent_position
            agent_loc = self.grid_world.get_agent_location()
            
            if agent_loc != obj_loc:
                # Need to move to the object's location first
                center_x, center_y = self.grid_world.location_centers[obj_loc]
                distance_to_loc = abs(agent_x - center_x) + abs(agent_y - center_y)
                
                # Move to the location center
                if self.grid_world.goto_location(obj_loc):
                    cost += distance_to_loc
                    agent_x, agent_y = center_x, center_y
            
            # Calculate distance to object's exact position
            distance_to_obj = abs(agent_x - obj_x) + abs(agent_y - obj_y)
            
            # Move to object's exact position (simulate)
            self.grid_world.agent_position = (obj_x, obj_y)
            cost += distance_to_obj
            
            # Perform pickup
            if self.grid_world.pickup_object(obj):
                # Remove object from knowledge base
                if obj in self.knowledge_base['objects']:
                    del self.knowledge_base['objects'][obj]
                self.held_object = obj
                self.update_knowledge()
                return True, cost + 1  # Add cost for pickup action
                
        elif cmd == "putdown" and len(parts) == 1:
            if self.held_object and self.grid_world.putdown_object(self.held_object):
                # Update knowledge with object's new position
                x, y = self.grid_world.agent_position
                loc = self.grid_world.get_agent_location()
                self.knowledge_base['objects'][self.held_object] = (x, y, loc)
                
                self.held_object = None
                self.update_knowledge()
                return True, 1  # Cost of putdown action
                
        elif cmd == "placenear" and len(parts) == 2:
            # For placenear, we need to be holding an object
            if not self.held_object:
                return False, 0
                
            target_obj = parts[1]
            # Check if target object is known
            if target_obj not in self.knowledge_base['objects']:
                return False, 0
                
            # Get target object's exact position and location
            target_x, target_y, target_loc = self.knowledge_base['objects'][target_obj]
            agent_x, agent_y = self.grid_world.agent_position
            agent_loc = self.grid_world.get_agent_location()
            
            if agent_loc != target_loc:
                # Need to move to the target object's location
                center_x, center_y = self.grid_world.location_centers[target_loc]
                distance_to_loc = abs(agent_x - center_x) + abs(agent_y - center_y)
                
                # Move to the location center
                if self.grid_world.goto_location(target_loc):
                    cost += distance_to_loc
                    agent_x, agent_y = center_x, center_y
            
            # Calculate distance to target object's exact position
            distance_to_target = abs(agent_x - target_x) + abs(agent_y - target_y)
            
            # Move to target object's exact position (simulate)
            self.grid_world.agent_position = (target_x, target_y)
            cost += distance_to_target
            
            # Place the held object near the target
            if self.grid_world.putdown_object(self.held_object):
                # Update knowledge with the object's new position
                x, y = self.grid_world.agent_position
                loc = self.grid_world.get_agent_location()
                self.knowledge_base['objects'][self.held_object] = (x, y, loc)
                
                # Add the near relationship to the knowledge base
                if 'near' not in self.knowledge_base:
                    self.knowledge_base['near'] = set()
                self.knowledge_base['near'].add((self.held_object, target_obj))
                self.knowledge_base['near'].add((target_obj, self.held_object))
                
                self.held_object = None
                self.update_knowledge()
                return True, cost + 1  # Add cost for placenear action
                        
        return False, 0

class SpatialGame:
    def __init__(self):
        self.grid_world = GridWorld()
        self.agent = Agent(self.grid_world)
        self.goal_formula = None
        
    def set_goal(self, goal_formula):
        self.goal_formula = goal_formula
    
    def automated_mission(self, max_steps=10000, planning_interval=100):
        print(f"Starting mission with goal: {self.goal_formula}")
        
        total_cost = 0
        plan = []
        
        while total_cost < max_steps:
            # Update agent's knowledge based on current perception
            self.agent.update_knowledge()
            
            # Check if goal is already achieved
            if self.agent.domain.evaluate_formula(self.agent.current_state, self.goal_formula):
                print(f"Goal achieved with total cost {total_cost}!")
                return True, total_cost
            
            # Check if we have a plan and execute it
            if plan:
                action = plan.pop(0)
                print(f"Executing: {action}")
                success, cost = self.agent.execute_action(action)
                total_cost += cost
                if success:
                    print(f"Action successful, cost: {cost}, total cost: {total_cost}")
                else:
                    print(f"Action failed, cost: {cost}, replanning...")
                    plan = []  # Clear plan if action fails
            else:
                # Check if we can find a plan periodically
                if total_cost % planning_interval == 0:
                    print("Attempting to find a plan...")
                    plan = self.agent.find_plan(self.goal_formula, print_planning = total_cost>850)
                    if plan:
                        print(f"Found plan: {plan}")
                    else:
                        print("No plan found, continuing exploration")
                
                # Continue exploration
                success, cost = self.agent.systematic_exploration()
                total_cost += cost
                if success:
                    print(f"Exploration step, cost: {cost}, total cost: {total_cost}")
            
            # Check if goal is achieved after action
            if self.agent.domain.evaluate_formula(self.agent.current_state, self.goal_formula):
                print(f"Goal achieved with total cost {total_cost}!")
                return True, total_cost
        
        print("Mission failed: timeout reached")
        return False, total_cost

# Example usage
if __name__ == "__main__":
    # Create game
    game = SpatialGame()
    
    # Set a goal formula (book near cup in location A)
    goal_formula = ('and',
        ('at', 'book', 'A'),
        ('at', 'cup', 'A'),
        ('near', 'book', 'cup')
    )
    game.set_goal(goal_formula)
    
    # Run the mission
    success, total_cost = game.automated_mission(max_steps=5000, planning_interval=50)
    print(f"Mission {'succeeded' if success else 'failed'} with total cost {total_cost}")