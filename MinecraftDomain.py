from typing import List, Dict, Set, Any, Optional, Tuple
from enum import Enum
from collections import deque

class Location(Enum):
    FOREST = "forest"
    MINE = "mine" 
    ORCHARD = "orchard"
    CRAFTING_TABLE = "crafting_table"
    VILLAGE = "village"
    
    def __lt__(self, other):
        return self.value < other.value

class Item(Enum):
    # Basic resources
    WOOD = "wood"
    STICKS = "sticks"
    STONE = "stone"
    IRON = "iron"
    DIAMONDS = "diamonds"
    STRING = "string"
    GOLD = "gold"
    APPLE = "apple"
    LAPIS = "lapis"
    MEAT = "meat"
    
    # Tools
    WOOD_PICKAXE = "wood_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
    
    # Weapons
    WOOD_SWORD = "wood_sword"
    STONE_SWORD = "stone_sword"
    DIAMOND_SWORD = "diamond_sword"
    ENCHANTED_BOW = "enchanted_bow"
    
    # Special
    GOLDEN_APPLE = "golden_apple"
    
    def __lt__(self, other):
        return self.value < other.value

class ActionType(Enum):
    GATHER = "gather"
    CRAFT = "craft"
    TRADE = "trade"
    USE_TOOL = "use_tool"
    MOVE = "move"

class MinecraftAction:
    def __init__(self, name: str, action_type: ActionType, location: Location,
                 prerequisites: Dict[Item, int], results: Dict[Item, int]):
        self.name = name
        self.action_type = action_type
        self.location = location
        self.prerequisites = prerequisites
        self.results = results
    
    def can_execute(self, inventory: Dict[Item, int], current_location: Location) -> bool:
        if self.action_type != ActionType.MOVE and current_location != self.location:
            return False
            
        for item, count in self.prerequisites.items():
            if item not in inventory or inventory[item] < count:
                return False
        return True
    
    def execute(self, inventory: Dict[Item, int]) -> Dict[Item, int]:
        new_inventory = inventory.copy()
        
        # Remove prerequisites (except for move actions)
        if self.action_type != ActionType.MOVE:
            for item, count in self.prerequisites.items():
                if item in new_inventory:
                    new_inventory[item] -= count
                    if new_inventory[item] <= 0:
                        del new_inventory[item]
        
        # Add results (except for move actions)
        if self.action_type != ActionType.MOVE:
            for item, count in self.results.items():
                new_inventory[item] = new_inventory.get(item, 0) + count
            
        return new_inventory
    
    def __str__(self):
        if self.action_type == ActionType.MOVE:
            return f"{self.name}"
        prereq_str = " + ".join([f"{count}x {item.value}" for item, count in self.prerequisites.items()])
        result_str = " + ".join([f"{count}x {item.value}" for item, count in self.results.items()])
        return f"{self.name:25} | {prereq_str:30} → {result_str}"

class MinecraftPlanningDomain:
    def __init__(self):
        # Define locations first
        self.locations = list(Location)
        self.location_connections = self._create_location_connections()
        self.actions = self._create_actions()
    
    def _create_location_connections(self) -> Dict[Location, List[Location]]:
        """Define which locations are connected to each other"""
        return {
            Location.FOREST: [Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.MINE: [Location.FOREST, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.ORCHARD: [Location.FOREST, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.CRAFTING_TABLE: [Location.FOREST, Location.MINE, Location.ORCHARD, Location.VILLAGE],
            Location.VILLAGE: [Location.FOREST, Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE],
        }
    
    def _create_actions(self) -> List[MinecraftAction]:
        actions = []
        
        # Movement actions 
        for from_loc in self.locations:
            for to_loc in self.locations:
                if from_loc != to_loc:
                    actions.append(
                        MinecraftAction(
                            f"move_{from_loc.value}_to_{to_loc.value}",
                            ActionType.MOVE,
                            from_loc,
                            {},
                            {}
                        )
                    )
        
        # Gathering actions
        actions.extend([
            MinecraftAction("gather_wood", ActionType.GATHER, Location.FOREST, {}, {Item.WOOD: 1}),
            MinecraftAction("gather_sticks", ActionType.GATHER, Location.FOREST, {}, {Item.STICKS: 1}),
            MinecraftAction("hunt_meat", ActionType.GATHER, Location.FOREST, 
                          {Item.WOOD_SWORD: 1}, {Item.MEAT: 1}),
            MinecraftAction("hunt_meat_stone", ActionType.GATHER, Location.FOREST,
                          {Item.STONE_SWORD: 1}, {Item.MEAT: 1}),
            MinecraftAction("gather_apple", ActionType.GATHER, Location.ORCHARD, {}, {Item.APPLE: 1}),
            MinecraftAction("gather_stone", ActionType.GATHER, Location.MINE, {}, {Item.STONE: 1}),
            MinecraftAction("gather_string", ActionType.GATHER, Location.MINE, {}, {Item.STRING: 1}),
            MinecraftAction("mine_iron", ActionType.USE_TOOL, Location.MINE,
                          {Item.WOOD_PICKAXE: 1}, {Item.IRON: 1}),
            MinecraftAction("mine_diamonds", ActionType.USE_TOOL, Location.MINE,
                          {Item.IRON_PICKAXE: 1}, {Item.DIAMONDS: 1}),
            MinecraftAction("mine_gold", ActionType.USE_TOOL, Location.MINE,
                          {Item.IRON_PICKAXE: 1}, {Item.GOLD: 1}),
            MinecraftAction("mine_lapis", ActionType.USE_TOOL, Location.MINE,
                          {Item.IRON_PICKAXE: 1}, {Item.LAPIS: 1}),
        ])
        
        # Crafting actions
        actions.extend([
            MinecraftAction("craft_wood_sword", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.WOOD: 1}, {Item.WOOD_SWORD: 1}),
            MinecraftAction("craft_stone_sword", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.STONE: 1}, {Item.STONE_SWORD: 1}),
            MinecraftAction("craft_wood_pickaxe", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.WOOD: 1, Item.STICKS: 1}, {Item.WOOD_PICKAXE: 1}),
            MinecraftAction("craft_iron_pickaxe", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.IRON: 1, Item.STICKS: 1}, {Item.IRON_PICKAXE: 1}),
            MinecraftAction("craft_diamond_sword", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.DIAMONDS: 1, Item.STICKS: 1}, {Item.DIAMOND_SWORD: 1}),
            MinecraftAction("craft_golden_apple", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.APPLE: 1, Item.GOLD: 1}, {Item.GOLDEN_APPLE: 1}),
            MinecraftAction("craft_enchanted_bow", ActionType.CRAFT, Location.CRAFTING_TABLE,
                          {Item.STRING: 1, Item.LAPIS: 1}, {Item.ENCHANTED_BOW: 1}),
        ])
        
        # Trading actions
        actions.extend([
            MinecraftAction("trade_wood_for_iron_pick", ActionType.TRADE, Location.VILLAGE,
                          {Item.WOOD: 1}, {Item.IRON_PICKAXE: 1}),
            MinecraftAction("trade_sticks_for_string", ActionType.TRADE, Location.VILLAGE,
                          {Item.STICKS: 1}, {Item.STRING: 1}),
            MinecraftAction("trade_stone_for_lapis", ActionType.TRADE, Location.VILLAGE,
                          {Item.STONE: 1}, {Item.LAPIS: 1}),
            MinecraftAction("trade_lapis_for_gold", ActionType.TRADE, Location.VILLAGE,
                          {Item.LAPIS: 1}, {Item.GOLD: 1}),
            MinecraftAction("trade_meat_for_diamonds", ActionType.TRADE, Location.VILLAGE,
                          {Item.MEAT: 1}, {Item.DIAMONDS: 1}),
        ])
        
        return actions
    
    def get_available_actions(self, inventory: Dict[Item, int], current_location: Location) -> List[MinecraftAction]:
        """Get all actions that can be executed in current state"""
        available = []
        for action in self.actions:
            if action.can_execute(inventory, current_location):
                available.append(action)
        return available
    
    def _get_state_signature(self, location: Location, inventory: Dict[Item, int]) -> Tuple:
        """Create a hashable state signature for cycle detection"""
        inventory_tuple = tuple(sorted((item.value, count) for item, count in inventory.items()))
        return (location.value, inventory_tuple)
    
    def find_shortest_path(self, start_location: Location, goal_items: Dict[Item, int],
                          max_depth: int = 20) -> Optional[List[MinecraftAction]]:
        """Find the shortest path using BFS"""
        initial_state = (start_location, {})
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue:
            (location, inventory), path = queue.popleft()
            
            # Check if goal is achieved
            if all(inventory.get(item, 0) >= count for item, count in goal_items.items()):
                return path
            
            # Skip if visited
            state_sig = self._get_state_signature(location, inventory)
            if state_sig in visited:
                continue
            visited.add(state_sig)
            
            # Check depth limit
            if len(path) >= max_depth:
                continue
            
            # Try all available actions
            for action in self.get_available_actions(inventory, location):
                new_inventory = action.execute(inventory)
                
                # For move actions, the new location is the target location
                if action.action_type == ActionType.MOVE:
                    # Extract target location from action name: "move_X_to_Y"
                    target_loc_name = action.name.split("_to_")[1]
                    new_location = Location(target_loc_name)
                else:
                    new_location = action.location
                
                new_path = path + [action]
                new_state = (new_location, new_inventory)
                queue.append((new_state, new_path))
        
        return None

# Test the domain
def test_basic_functionality():
    """Test that the domain works without errors"""
    print("=== Testing Basic Domain Functionality ===")
    
    try:
        domain = MinecraftPlanningDomain()
        print("✅ Domain created successfully")
        print(f"✅ Locations: {[loc.value for loc in domain.locations]}")
        print(f"✅ Total actions: {len(domain.actions)}")
        
        # Test available actions from forest
        actions = domain.get_available_actions({}, Location.FOREST)
        print(f"✅ Available actions from Forest: {len(actions)}")
        
        # Count by type
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        gather_actions = [a for a in actions if a.action_type == ActionType.GATHER]
        
        print(f"✅ Move actions: {len(move_actions)}")
        print(f"✅ Gather actions: {len(gather_actions)}")
        
        return domain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_simple_paths(domain):
    """Test finding simple paths"""
    print("\n=== Testing Simple Paths ===")
    
    # Test 1: Wood Sword (should be simple: gather wood, move to crafting, craft)
    print("\n🎯 Test 1: Path to Wood Sword")
    path = domain.find_shortest_path(Location.FOREST, {Item.WOOD_SWORD: 1}, max_depth=10)
    
    if path:
        print(f"✅ Success! {len(path)} steps:")
        for i, action in enumerate(path, 1):
            if action.action_type == ActionType.MOVE:
                print(f"  {i:2}. 🚶 {action}")
            else:
                location_icon = {
                    Location.FOREST: "🌲",
                    Location.MINE: "⛏️", 
                    Location.ORCHARD: "🍎",
                    Location.CRAFTING_TABLE: "🛠️",
                    Location.VILLAGE: "🏘️"
                }.get(action.location, "📍")
                print(f"  {i:2}. {location_icon} {action}")
    else:
        print("❌ Failed to find path to Wood Sword")
    
    # Test 2: Wood Pickaxe
    print("\n🎯 Test 2: Path to Wood Pickaxe")
    path = domain.find_shortest_path(Location.FOREST, {Item.WOOD_PICKAXE: 1}, max_depth=10)
    
    if path:
        print(f"✅ Success! {len(path)} steps:")
        for i, action in enumerate(path, 1):
            if action.action_type == ActionType.MOVE:
                print(f"  {i:2}. 🚶 {action}")
            else:
                location_icon = {
                    Location.FOREST: "🌲",
                    Location.MINE: "⛏️",
                    Location.ORCHARD: "🍎",
                    Location.CRAFTING_TABLE: "🛠️", 
                    Location.VILLAGE: "🏘️"
                }.get(action.location, "📍")
                print(f"  {i:2}. {location_icon} {action}")
    else:
        print("❌ Failed to find path to Wood Pickaxe")

def analyze_domain(domain):
    """Analyze the domain structure"""
    print("\n=== Domain Analysis ===")
    
    action_types = {}
    for action in domain.actions:
        action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
    
    print("Action types:")
    for action_type, count in action_types.items():
        print(f"  {action_type.value}: {count}")
    
    print(f"\nLocations: {len(domain.locations)}")
    for location in domain.locations:
        actions_at_loc = [a for a in domain.actions if a.location == location and a.action_type != ActionType.MOVE]
        print(f"  {location.value}: {len(actions_at_loc)} actions")

if __name__ == "__main__":
    domain = test_basic_functionality()
    if domain:
        analyze_domain(domain)
        test_simple_paths(domain)