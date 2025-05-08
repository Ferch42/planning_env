from dataclasses import dataclass, replace
from typing import Set, FrozenSet, Dict, List, Optional
from collections import deque

@dataclass(frozen=True)
class State:
    holding: FrozenSet[str]
    available: FrozenSet[str]
    
    def __repr__(self):
        return f"Holding: {set(self.holding)}, Available: {set(self.available)}"

class CraftingDomain:
    def __init__(self):
        self.RECIPES: Dict[str, FrozenSet[str]] = {
            'Basic_Engine': frozenset({'Iron', 'Fuel'}),
            'Thermal_Core': frozenset({'Copper', 'Stone'}),
            'Hybrid_Drive': frozenset({'Basic_Engine', 'Thermal_Core'}),
            'Aerial_Transport': frozenset({'Hybrid_Drive', 'Wood'}),
            'Reinforced_Frame': frozenset({'Basic_Engine', 'Wood'})
        }
        self.DECOMPOSITIONS = {v: k for k, v in self.RECIPES.items()}
        self.BASIC_ITEMS = {'Iron', 'Fuel', 'Copper', 'Stone', 'Wood'}
        self.MAX_CAPACITY = 3

    def get_components(self, item: str) -> Optional[FrozenSet[str]]:
        return self.RECIPES.get(item, None)
    
    def get_decomposition(self, item: str) -> Optional[FrozenSet[str]]:
        return self.DECOMPOSITIONS.get(item, None)

    def hold(self, state: State, item: str) -> Optional[State]:
        if (len(state.holding) < self.MAX_CAPACITY and 
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
        initial_state = State(
            holding=frozenset(),
            available=frozenset(self.BASIC_ITEMS)
        )
        
        queue = deque([(initial_state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()
            
            # Check goal condition
            if goal in current_state.holding:
                return path
            
            # State signature for cycle detection
            state_sig = (current_state.holding, current_state.available)
            if state_sig in visited:
                continue
            visited.add(state_sig)

            # Generate possible actions
            # 1. Hold actions
            for item in current_state.available:
                if new_state := self.hold(current_state, item):
                    queue.append((new_state, path + [f"hold {item}"]))
            
            # 2. Drop actions
            for item in current_state.holding:
                if new_state := self.drop(current_state, item):
                    queue.append((new_state, path + [f"drop {item}"]))
            
            # 3. Build actions
            for target in self.RECIPES:
                if new_state := self.build(current_state, target):
                    queue.append((new_state, path + [f"build {target}"]))
            
            # 4. Decompose actions
            for item in current_state.holding:
                if new_state := self.decompose(current_state, item):
                    queue.append((new_state, path + [f"decompose {item}"]))
        
        return []  # No plan found

if __name__ == "__main__":
    domain = CraftingDomain()
    
    # Find plan for Hybrid_Drive OR Aerial_Transport
    plan = None
    for target in ['Aerial_Transport']:
        if plan := domain.find_plan(target):
            print(f"Plan to build {target}:")
            for step in plan:
                print(f"  {step}")
            break
    
    if not plan:
        print("No valid plan found")