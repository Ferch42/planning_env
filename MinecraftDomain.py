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
    def __init__(self, available_locations: Set[Location] = None):
        # Use provided locations or default to all locations
        if available_locations is None:
            available_locations = set(Location)
        self.available_locations = available_locations
        self.location_connections = self._create_location_connections()
        self.actions = self._create_actions()
    
    def _create_location_connections(self) -> Dict[Location, List[Location]]:
        """Define which locations are connected to each other, filtered by available locations"""
        # Define all possible connections
        all_connections = {
            Location.FOREST: [Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.MINE: [Location.FOREST, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.ORCHARD: [Location.FOREST, Location.CRAFTING_TABLE, Location.VILLAGE],
            Location.CRAFTING_TABLE: [Location.FOREST, Location.MINE, Location.ORCHARD, Location.VILLAGE],
            Location.VILLAGE: [Location.FOREST, Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE],
        }
        
        # Filter connections to only include available locations
        filtered_connections = {}
        for location, connected_locations in all_connections.items():
            if location in self.available_locations:
                # Only keep connections to locations that are available
                filtered_connections[location] = [
                    connected_loc for connected_loc in connected_locations 
                    if connected_loc in self.available_locations
                ]
        
        return filtered_connections
    
    def _create_actions(self) -> List[MinecraftAction]:
        actions = []
        
        # Movement actions 
        for from_loc in self.available_locationscl:
            for to_loc in self.available_locationscl:
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


import unittest
from typing import Set, Dict, List

class TestMinecraftPlanningDomain(unittest.TestCase):
    
    def test_default_constructor(self):
        """Test that default constructor works and uses all locations"""
        domain = MinecraftPlanningDomain()
        self.assertEqual(domain.available_locations, set(Location))
        self.assertEqual(len(domain.location_connections), 5)  # All 5 locations should have connections
    
    def test_custom_locations_forest_mine_only(self):
        """Test with only Forest and Mine available"""
        available_locations = {Location.FOREST, Location.MINE}
        domain = MinecraftPlanningDomain(available_locations)
        
        self.assertEqual(domain.available_locations, available_locations)
        
        # Check connections only include available locations
        self.assertEqual(set(domain.location_connections.keys()), available_locations)
        
        # Forest should only connect to Mine (other connections removed)
        self.assertEqual(domain.location_connections[Location.FOREST], [Location.MINE])
        
        # Mine should only connect to Forest
        self.assertEqual(domain.location_connections[Location.MINE], [Location.FOREST])
    
    def test_custom_locations_no_village(self):
        """Test world without village"""
        available_locations = {Location.FOREST, Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE}
        domain = MinecraftPlanningDomain(available_locations)
        
        self.assertEqual(domain.available_locations, available_locations)
        
        # Village should not be in connections
        self.assertNotIn(Location.VILLAGE, domain.location_connections)
        
        # Check Forest connections (no village)
        forest_connections = domain.location_connections[Location.FOREST]
        self.assertIn(Location.MINE, forest_connections)
        self.assertIn(Location.ORCHARD, forest_connections)
        self.assertIn(Location.CRAFTING_TABLE, forest_connections)
        self.assertNotIn(Location.VILLAGE, forest_connections)
    
    def test_custom_locations_trading_world(self):
        """Test world with only Forest and Village (trading-focused)"""
        available_locations = {Location.FOREST, Location.VILLAGE}
        domain = MinecraftPlanningDomain(available_locations)
        
        self.assertEqual(domain.available_locations, available_locations)
        
        # Both should connect to each other
        self.assertEqual(domain.location_connections[Location.FOREST], [Location.VILLAGE])
        self.assertEqual(domain.location_connections[Location.VILLAGE], [Location.FOREST])
    
    def test_custom_locations_single_location(self):
        """Test with only one location available"""
        available_locations = {Location.FOREST}
        domain = MinecraftPlanningDomain(available_locations)
        
        self.assertEqual(domain.available_locations, available_locations)
        
        # Forest should have no connections (only location)
        self.assertEqual(domain.location_connections[Location.FOREST], [])
    
    def test_empty_locations(self):
        """Test with empty set of locations"""
        available_locations = set()
        domain = MinecraftPlanningDomain(available_locations)
        
        self.assertEqual(domain.available_locations, set())
        self.assertEqual(domain.location_connections, {})
    
    def test_movement_actions_respect_available_locations(self):
        """Test that movement actions are only created for available locations"""
        available_locations = {Location.FOREST, Location.MINE}
        domain = MinecraftPlanningDomain(available_locations)
        
        # Count movement actions - should only be between Forest and Mine
        move_actions = [a for a in domain.actions if a.action_type == ActionType.MOVE]
        
        # Should have 2 movement actions: forest->mine and mine->forest
        self.assertEqual(len(move_actions), 2)
        
        move_action_names = [a.name for a in move_actions]
        self.assertIn("move_forest_to_mine", move_action_names)
        self.assertIn("move_mine_to_forest", move_action_names)
        
        # Should NOT have actions to unavailable locations
        unavailable_moves = [name for name in move_action_names 
                           if "orchard" in name or "village" in name or "crafting_table" in name]
        self.assertEqual(len(unavailable_moves), 0)
    
    def test_gathering_actions_respect_available_locations(self):
        """Test that gathering actions are only created for available locations"""
        available_locations = {Location.FOREST}  # Only forest, no mine
        domain = MinecraftPlanningDomain(available_locations)
        
        # Should have forest gathering actions
        forest_actions = [a for a in domain.actions 
                         if a.location == Location.FOREST and a.action_type == ActionType.GATHER]
        self.assertGreater(len(forest_actions), 0)
        
        # Should NOT have mine gathering actions
        mine_actions = [a for a in domain.actions 
                       if a.location == Location.MINE and a.action_type == ActionType.GATHER]
        self.assertEqual(len(mine_actions), 0)
    
    def test_crafting_actions_only_with_crafting_table(self):
        """Test that crafting actions are only available when crafting table exists"""
        # Test without crafting table
        domain_no_crafting = MinecraftPlanningDomain({Location.FOREST, Location.MINE})
        crafting_actions = [a for a in domain_no_crafting.actions if a.action_type == ActionType.CRAFT]
        self.assertEqual(len(crafting_actions), 0)
        
        # Test with crafting table
        domain_with_crafting = MinecraftPlanningDomain({Location.FOREST, Location.CRAFTING_TABLE})
        crafting_actions = [a for a in domain_with_crafting.actions if a.action_type == ActionType.CRAFT]
        self.assertGreater(len(crafting_actions), 0)
    
    def test_trading_actions_only_with_village(self):
        """Test that trading actions are only available when village exists"""
        # Test without village
        domain_no_village = MinecraftPlanningDomain({Location.FOREST, Location.MINE})
        trading_actions = [a for a in domain_no_village.actions if a.action_type == ActionType.TRADE]
        self.assertEqual(len(trading_actions), 0)
        
        # Test with village
        domain_with_village = MinecraftPlanningDomain({Location.FOREST, Location.VILLAGE})
        trading_actions = [a for a in domain_with_village.actions if a.action_type == ActionType.TRADE]
        self.assertGreater(len(trading_actions), 0)
    
    def test_pathfinding_respects_available_locations(self):
        """Test that pathfinding works correctly with limited locations"""
        # World with only forest and crafting table
        domain = MinecraftPlanningDomain({Location.FOREST, Location.CRAFTING_TABLE})
        
        # Should be able to make wood sword
        path = domain.find_shortest_path(Location.FOREST, {Item.WOOD_SWORD: 1}, max_depth=10)
        self.assertIsNotNone(path)
        
        # Should NOT be able to make iron pickaxe (no mine or village)
        path = domain.find_shortest_path(Location.FOREST, {Item.IRON_PICKAXE: 1}, max_depth=10)
        self.assertIsNone(path)
    
    def test_backward_compatibility(self):
        """Test that original behavior is preserved when no locations are specified"""
        domain_old = MinecraftPlanningDomain()  # No parameters
        domain_new = MinecraftPlanningDomain(set(Location))  # Explicit all locations
        
        # Both should have the same available locations
        self.assertEqual(domain_old.available_locations, domain_new.available_locations)
        
        # Both should have the same number of actions
        self.assertEqual(len(domain_old.actions), len(domain_new.actions))
        
        # Both should find the same paths
        path_old = domain_old.find_shortest_path(Location.FOREST, {Item.WOOD_SWORD: 1}, max_depth=10)
        path_new = domain_new.find_shortest_path(Location.FOREST, {Item.WOOD_SWORD: 1}, max_depth=10)
        self.assertEqual(len(path_old), len(path_new))

def run_comprehensive_demonstration():
    """Run a comprehensive demonstration of the new functionality"""
    print("🧪 COMPREHENSIVE TEST DEMONSTRATION")
    print("=" * 60)
    
    test_scenarios = [
        {
            "name": "Complete World",
            "locations": set(Location),
            "tests": ["All locations available", "Full connectivity", "All action types"]
        },
        {
            "name": "No Village World", 
            "locations": {Location.FOREST, Location.MINE, Location.ORCHARD, Location.CRAFTING_TABLE},
            "tests": ["No trading", "Traditional crafting only", "No village connections"]
        },
        {
            "name": "Forest Only",
            "locations": {Location.FOREST},
            "tests": ["Limited gathering", "No crafting/trading", "No movement options"]
        },
        {
            "name": "Trading World",
            "locations": {Location.FOREST, Location.VILLAGE},
            "tests": ["Optimal trading", "No mining", "Quick tool acquisition"]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🌍 {scenario['name']}")
        print(f"   Locations: {[loc.value for loc in scenario['locations']]}")
        print(f"   Tests: {', '.join(scenario['tests'])}")
        
        domain = MinecraftPlanningDomain(scenario['locations'])
        
        # Analyze the domain
        action_types = {}
        for action in domain.actions:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        
        print(f"   Action counts: {action_types}")
        print(f"   Connections: {len(domain.location_connections)} locations connected")
        
        # Test basic functionality
        goals_to_test = [
            ("Wood Sword", {Item.WOOD_SWORD: 1}),
            ("Iron Pickaxe", {Item.IRON_PICKAXE: 1}),
        ]
        
        for goal_name, goal_items in goals_to_test:
            path = domain.find_shortest_path(Location.FOREST, goal_items, max_depth=15)
            if path:
                print(f"   ✅ {goal_name}: {len(path)} steps")
            else:
                print(f"   ❌ {goal_name}: Not achievable")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    run_comprehensive_demonstration()
    
    print("\n" + "=" * 60)
    print("🧪 RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Run the unit tests
    unittest.main(argv=[''], verbosity=2, exit=False)