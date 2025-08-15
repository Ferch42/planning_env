from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple, Any, Dict
from collections import deque

@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Tuple]  # First-order logic predicates

class SpatialDomain:
    VALID_ACTIONS = {'goto', 'pickup', 'putdown', 'placenear'}
    
    def __init__(self, locations: Set[str], objects: Set[str], connections: Set[Tuple[str, str]]):
        self.locations = locations
        self.objects = objects
        self.connections = connections  # Bidirectional location connections
        # Create adjacency list for efficient movement
        self.adjacency: Dict[str, Set[str]] = {loc: set() for loc in locations}
        for loc1, loc2 in connections:
            self.adjacency[loc1].add(loc2)
            self.adjacency[loc2].add(loc1)

    # ====== Predicate Evaluation Methods ======
    def get_agent_location(self, state: State) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == 'agent_at' and len(pred) == 2:
                return pred[1]
        return None

    def is_holding(self, state: State) -> bool:
        """Check if agent is holding any object (only one allowed)"""
        return any(p[0] == 'holding' for p in state.predicates)
    
    def get_held_object(self, state: State) -> Optional[str]:
        """Get the object being held, if any"""
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
        """Check if two objects are near each other (explicit relationship)"""
        return (('near', obj1, obj2) in state.predicates) or \
               (('near', obj2, obj1) in state.predicates)

    # ====== Action Implementation ======
    def goto(self, state: State, new_loc: str) -> Optional[State]:
        current_loc = self.get_agent_location(state)
        if current_loc is None or new_loc not in self.adjacency.get(current_loc, set()):
            return None  # Invalid move
            
        new_predicates = set(state.predicates)
        # Remove current agent location
        new_predicates.discard(('agent_at', current_loc))
        # Add new agent location
        new_predicates.add(('agent_at', new_loc))
        return State(frozenset(new_predicates))

    def pickup(self, state: State, obj: str) -> Optional[State]:
        # Check if agent is already holding something
        if self.is_holding(state):
            return None
            
        # Check if object is available at agent's location
        agent_loc = self.get_agent_location(state)
        obj_loc = self.get_object_location(state, obj)
        
        if agent_loc is None or obj_loc != agent_loc:
            return None  # Object not at agent's location
            
        new_predicates = set(state.predicates)
        # Remove location predicate
        new_predicates.discard(('at', obj, obj_loc))
        # Add holding predicate
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
        # Remove holding predicate
        new_predicates.discard(('holding', obj))
        # Add location predicate at agent's current location
        new_predicates.add(('at', obj, agent_loc))
        return State(frozenset(new_predicates))

    def placenear(self, state: State, target_obj: str) -> Optional[State]:
        # Get held object
        held_obj = self.get_held_object(state)
        if held_obj is None:
            return None
            
        # Target object must be placed
        target_loc = self.get_object_location(state, target_obj)
        if target_loc is None:
            return None
            
        # Both objects must be at the same location
        agent_loc = self.get_agent_location(state)
        if agent_loc != target_loc:
            return None
            
        new_predicates = set(state.predicates)
        # Remove holding predicate
        new_predicates.discard(('holding', held_obj))
        # Add location predicate at agent's current location
        new_predicates.add(('at', held_obj, agent_loc))
        # Add near relationship
        new_predicates.add(('near', held_obj, target_obj))
        return State(frozenset(new_predicates))

    # ====== Logical Formula Evaluation ======
    def evaluate_formula(self, state: State, formula: Any) -> bool:
        """Evaluate a logical formula against the current state"""
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
                # Check if holding a specific object
                if len(formula) == 1:
                    return self.is_holding(state)
                return self.get_held_object(state) == formula[1]
                
            elif operator == 'at':
                return self.get_object_location(state, formula[1]) == formula[2]
                
            elif operator == 'near':
                return self.are_near(state, formula[1], formula[2])
                
            # Handle atomic predicates directly
            return formula in state.predicates
            
        # Handle atomic predicates specified as strings
        return formula in state.predicates

    # ====== Planning Algorithm ======
    def find_plan(
        self,
        initial_state: State,
        goal_formula: Any,  # Logical formula defining the goal
        max_depth: int = 1000
    ) -> List[str]:
        queue = deque([(initial_state, [])])
        visited = set()
        visited.add(initial_state)
        
        while queue:
            state, path = queue.popleft()
            
            if self.evaluate_formula(state, goal_formula):
                return path
                
            if len(path) >= max_depth:
                continue
                
            for action in self._generate_actions(state):
                new_state = self.execute(state, action)
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action]))
        return []

    def _generate_actions(self, state: State) -> List[str]:
        actions = []
        agent_loc = self.get_agent_location(state)
        
        # Movement actions
        if agent_loc:
            for neighbor in self.adjacency.get(agent_loc, []):
                actions.append(f"goto {neighbor}")
        
        # Pickup actions - only if not holding anything
        if not self.is_holding(state):
            for obj in self.objects:
                if self.get_object_location(state, obj) == agent_loc:
                    actions.append(f"pickup {obj}")
        
        # Putdown action - only if holding something
        if self.is_holding(state):
            actions.append("putdown")
        
        # Placenear actions - only if holding something and target is in same room
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

if __name__=='__main__':

    # Define world geometry
    locations = {"kitchen", "living_room", "bedroom"}
    connections = {("kitchen", "living_room"), ("living_room", "bedroom")}
    objects = {"book", "cup", "apple"}

    # Initial state: agent in kitchen, objects in living room
    initial_predicates = {
        ('agent_at', 'kitchen'),
        ('at', 'book', 'living_room'),
        ('at', 'cup', 'living_room'),
        ('at', 'apple', 'living_room')
    }
    initial_state = State(frozenset(initial_predicates))

    # Goal: Book near cup in the bedroom
    goal = ('or',
        ('and',
            ('near', 'book', 'cup'),
            ('at', 'book', 'bedroom'),
            ('at', 'cup', 'bedroom')
        ),
        ('and',
            ('near', 'apple', 'cup'),
            ('at', 'apple', 'kitchen'),
            ('at', 'cup', 'kitchen')
        )
    )

    # Create domain and find plan
    domain = SpatialDomain(locations, objects, connections)
    plan = domain.find_plan(initial_state, goal)
    print("Plan:", plan)