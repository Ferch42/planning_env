from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple, Any, Dict
from collections import deque

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
        if new_loc not in self.locations:
            return None  # Invalid location
            
        current_loc = self.get_agent_location(state)
        if current_loc == new_loc:
            return None  # Already at location
            
        new_predicates = set(state.predicates)
        if current_loc:
            new_predicates.discard(('agent_at', current_loc))
        new_predicates.add(('agent_at', new_loc))
        return State(frozenset(new_predicates))

    def pickup(self, state: State, obj: str) -> Optional[State]:
        if self.is_holding(state):
            return None  # Already holding something
            
        agent_loc = self.get_agent_location(state)
        obj_loc = self.get_object_location(state, obj)
        
        if agent_loc is None or obj_loc != agent_loc:
            return None  # Object not at agent's location
            
        new_predicates = set(state.predicates)
        new_predicates.discard(('at', obj, obj_loc))
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
        
        # Movement actions - can go to any location except current
        for loc in self.locations:
            if loc != agent_loc:
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

if __name__=='__main__':

    # Define world - no connections needed
    locations = {"kitchen", "living_room", "bedroom"}
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
    goal_formula = ('and',
        ('agent_at', 'bedroom'),
        ('near', 'book', 'cup'),
        ('at', 'book', 'bedroom'),
        ('at', 'cup', 'bedroom')
    )

    # Create domain
    domain = SpatialDomain(locations, objects)
    plan = domain.find_plan(initial_state, goal_formula)
    print("Plan:", plan)