from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Collection
from collections import deque

@dataclass(frozen=True)
class State:
    holding: FrozenSet[str]
    available: FrozenSet[str]
    
    def __repr__(self):
        return f"Holding: {set(self.holding)}, Available: {set(self.available)}"

class CraftingDomain:
    VALID_ACTIONS = {'hold', 'drop', 'build', 'decompose'}
    
    def __init__(self, 
                 allowed_actions: Collection[str] = VALID_ACTIONS,
                 allowed_items: Collection[str] = None):
        # Recipe configuration
        self.RECIPES: Dict[str, FrozenSet[str]] = {
            'Basic_Engine': frozenset({'Iron', 'Fuel'}),
            'Thermal_Core': frozenset({'Copper', 'Stone'}),
            'Hybrid_Drive': frozenset({'Basic_Engine', 'Thermal_Core'}),
            'Aerial_Transport': frozenset({'Hybrid_Drive', 'Wood'}),
            'Reinforced_Frame': frozenset({'Basic_Engine', 'Wood'})
        }
        self.DECOMPOSITIONS = {v: k for k, v in self.RECIPES.items()}
        
        # Item configuration
        self.BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
        self.CRAFTED_ITEMS = set(self.RECIPES.keys())
        self.MAX_CAPACITY = 3
        
        # Validate actions
        invalid_actions = set(allowed_actions) - self.VALID_ACTIONS
        if invalid_actions:
            raise ValueError(f"Invalid actions: {invalid_actions}")
        self.allowed_actions = set(allowed_actions)
        
        # Set allowed items
        default_items = self.BASIC_ITEMS | self.CRAFTED_ITEMS
        self.allowed_items = set(allowed_items) if allowed_items else default_items

    def get_components(self, item: str) -> Optional[FrozenSet[str]]:
        return self.RECIPES.get(item, None)
    
    def get_decomposition(self, item: str) -> Optional[FrozenSet[str]]:
        return self.DECOMPOSITIONS.get(item, None)

    def hold(self, state: State, item: str) -> Optional[State]:
        if (item in self.allowed_items and
            len(state.holding) < self.MAX_CAPACITY and 
            item in state.available and
            item not in state.holding):
            
            new_holding = set(state.holding)
            new_holding.add(item)
            new_available = set(state.available)
            new_available.remove(item)
            return State(frozenset(new_holding), frozenset(new_available))
        return None

    def drop(self, state: State, item: str) -> Optional[State]:
        if item in state.holding:
            new_holding = set(state.holding)
            new_holding.remove(item)
            new_available = set(state.available)
            new_available.add(item)
            return State(frozenset(new_holding), frozenset(new_available))
        return None

    def build(self, state: State, target: str) -> Optional[State]:
        components = self.get_components(target)
        if (components and 
            components.issubset(state.holding) and
            (len(state.holding) - len(components) + 1) <= self.MAX_CAPACITY):
            
            new_holding = set(state.holding)
            new_holding -= components
            new_holding.add(target)
            return State(frozenset(new_holding), state.available)
        return None

    def decompose(self, state: State, item: str) -> Optional[State]:
        components = self.get_decomposition(item)
        if (components and 
            item in state.holding and
            (len(state.holding) - 1 + len(components)) <= self.MAX_CAPACITY):
            
            new_holding = set(state.holding)
            new_holding.remove(item)
            new_holding.update(components)
            return State(frozenset(new_holding), state.available)
        return None

    def find_plan(self, goal: str) -> List[str]:
        # Initialize available items with allowed basic items
        initial_available = self.BASIC_ITEMS & self.allowed_items
        initial_state = State(
            holding=frozenset(),
            available=frozenset(initial_available)
        )
        
        queue = deque([(initial_state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()
            
            if goal in current_state.holding:
                return path
            
            state_sig = (current_state.holding, current_state.available)
            if state_sig in visited:
                continue
            visited.add(state_sig)

            # Generate possible actions
            if 'hold' in self.allowed_actions:
                for item in current_state.available:
                    if item in self.allowed_items:
                        if new_state := self.hold(current_state, item):
                            queue.append((new_state, path + [f"hold {item}"]))
            
            if 'drop' in self.allowed_actions:
                for item in current_state.holding:
                    if new_state := self.drop(current_state, item):
                        queue.append((new_state, path + [f"drop {item}"]))
            
            if 'build' in self.allowed_actions:
                for target in self.RECIPES:
                    if target in self.allowed_items:
                        if new_state := self.build(current_state, target):
                            queue.append((new_state, path + [f"build {target}"]))
            
            if 'decompose' in self.allowed_actions:
                for item in current_state.holding:
                    if item in self.allowed_items:
                        if new_state := self.decompose(current_state, item):
                            queue.append((new_state, path + [f"decompose {item}"]))
        
        return []

if __name__ == "__main__":
    # Example: Can only hold metal items and basic components
    domain = CraftingDomain(
       #allowed_actions=['build'],
       #allowed_items={'Copper'}
    )
    
    print("Trying to build Hybrid_Drive with restricted items:")
    plan = domain.find_plan('Hybrid_Drive')
    
    if plan:
        for step in plan:
            print(f"  {step}")
    else:
        print("No valid plan found with current restrictions")