from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Set, Tuple
from collections import deque

@dataclass(frozen=True)
class State:
    # Predicates: (predicate_name, args...)
    # Holding: ('holding', obj)
    # Location: ('at', obj, location)
    # Near: ('near', obj1, obj2)
    predicates: FrozenSet[Tuple]

class SpatialDomain:
    VALID_ACTIONS = {'pickup', 'putdown', 'placenear'}
    
    def __init__(self, locations: Set[str], objects: Set[str]):
        self.locations = locations
        self.objects = objects
        self.table_location = "table"  # Default location for placed objects

    def is_holding(self, state: State, obj: str) -> bool:
        return ('holding', obj) in state.predicates

    def get_object_location(self, state: State, obj: str) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == 'at' and pred[1] == obj:
                return pred[2]
        return None

    def are_near(self, state: State, obj1: str, obj2: str) -> bool:
        return (('near', obj1, obj2) in state.predicates) or \
               (('near', obj2, obj1) in state.predicates)

    def pickup(self, state: State, obj: str) -> Optional[State]:
        # Can only pickup if not holding anything and object is on table
        if any(p[0] == 'holding' for p in state.predicates):
            return None  # Already holding something
            
        obj_loc = self.get_object_location(state, obj)
        if obj_loc != self.table_location:
            return None  # Object not in valid location
            
        new_predicates = set(state.predicates)
        # Remove location predicate
        new_predicates.discard(('at', obj, self.table_location))
        # Add holding predicate
        new_predicates.add(('holding', obj))
        return State(frozenset(new_predicates))

    def putdown(self, state: State, obj: str) -> Optional[State]:
        if not self.is_holding(state, obj):
            return None  # Not holding this object
            
        new_predicates = set(state.predicates)
        # Remove holding predicate
        new_predicates.discard(('holding', obj))
        # Add location predicate
        new_predicates.add(('at', obj, self.table_location))
        return State(frozenset(new_predicates))

    def placenear(self, state: State, held_obj: str, target_obj: str) -> Optional[State]:
        # Must be holding the first object, and target must be placed
        if not self.is_holding(state, held_obj):
            return None
            
        if self.get_object_location(state, target_obj) is None:
            return None  # Target object not placed
            
        new_predicates = set(state.predicates)
        # Remove holding predicate
        new_predicates.discard(('holding', held_obj))
        # Add location predicate (placed at same table)
        new_predicates.add(('at', held_obj, self.table_location))
        # Add near relationship (symmetric)
        new_predicates.add(('near', held_obj, target_obj))
        new_predicates.add(('near', target_obj, held_obj))
        return State(frozenset(new_predicates))

    def find_plan(
        self,
        initial_state: State,
        goal_condition: callable,
        max_depth: int = 1000
    ) -> List[str]:
        queue = deque([(initial_state, [])])
        visited = set()
        visited.add(initial_state)
        
        while queue:
            state, path = queue.popleft()
            
            if goal_condition(state):
                return path
                
            if len(path) >= max_depth:
                continue
                
            # Generate all possible actions
            for action in self._generate_actions(state):
                new_state = self.execute(state, action)
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action]))
        return []

    def _generate_actions(self, state: State) -> List[str]:
        actions = []
        
        # Pickup actions
        for obj in self.objects:
            if self.get_object_location(state, obj) == self.table_location:
                actions.append(f"pickup {obj}")
        
        # Putdown actions (for any held object)
        for obj in self.objects:
            if self.is_holding(state, obj):
                actions.append(f"putdown {obj}")
        
        # Placenear actions
        for held_obj in self.objects:
            if self.is_holding(state, held_obj):
                for target_obj in self.objects:
                    if (held_obj != target_obj and 
                        self.get_object_location(state, target_obj) is not None):
                        actions.append(f"placenear {held_obj} {target_obj}")
                    
        return actions

    def execute(self, state: State, action: str) -> Optional[State]:
        parts = action.split()
        if not parts:
            return None
            
        cmd = parts[0]
        if cmd == "pickup" and len(parts) == 2:
            return self.pickup(state, parts[1])
        elif cmd == "putdown" and len(parts) == 2:
            return self.putdown(state, parts[1])
        elif cmd == "placenear" and len(parts) == 3:
            return self.placenear(state, parts[1], parts[2])
        return None
    

if __name__=='__main__':

    # Define objects and locations
    objects = {"book", "cup", "apple"}
    locations = {"table", "shelf"}  # 'table' is the default placement location

    # Initial state: all objects on table
    initial_predicates = {
        ('at', 'book', 'table'),
        ('at', 'cup', 'table'),
        ('at', 'apple', 'table')
    }
    initial_state = State(frozenset(initial_predicates))

    # Goal: Book near cup, and apple near cup
    def goal_condition(state):
        book_near_cup = ('near', 'book', 'cup') in state.predicates
        apple_near_cup = ('near', 'apple', 'cup') in state.predicates
        return book_near_cup and apple_near_cup

    # Create domain and find plan
    domain = SpatialDomain(locations, objects)
    plan = domain.find_plan(initial_state, goal_condition)
    print("Plan:", plan)