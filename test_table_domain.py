def test_door_transitions():
    """Test that door transitions correctly move the agent between rooms"""
    print("=== Testing Door Transitions ===")
    
    # Create a smaller grid for easier testing
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Get door transitions
    transitions = world.get_important_transitions()
    door_trans = transitions['door_transitions']
    
    print(f"Found {len(door_trans)} door transitions")
    
    # Test a few door transitions
    test_cases = door_trans[:10]  # Test first 4 door transitions
    
    for i, trans in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Transition: {trans['prev_position']} -> {trans['next_position']} via action {trans['action']}")
        
        # Set agent to starting position
        world.agent_pos = trans['prev_position']
        start_room = world.get_current_room_id()
        print(f"  Start position: {world.agent_pos}, Room: {start_room}")
        
        # Execute the action
        world.step(trans['action'])
        end_room = world.get_current_room_id()
        print(f"  End position: {world.agent_pos}, Room: {end_room}")
        
        # Verify the transition worked
        expected_pos = trans['next_position']
        if world.agent_pos == expected_pos:
            print("  ✓ SUCCESS: Agent moved to expected position")
        else:
            print(f"  ✗ FAILURE: Expected {expected_pos}, got {world.agent_pos}")
        
        if start_room != end_room:
            print("  ✓ SUCCESS: Agent changed rooms")
        else:
            # Check if this was supposed to be a room change
            prev_room = world.room_ids[trans['prev_position']]
            next_room = world.room_ids[trans['next_position']]
            if prev_room != next_room:
                print("  ✗ FAILURE: Agent should have changed rooms but didn't")
            else:
                print("  ✓ SUCCESS: No room change expected")


def test_object_transitions():
    """Test that object transitions correctly pick up and put down objects"""
    print("\n=== Testing Object Transitions ===")
    
    # Create a smaller grid for easier testing
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Get object transitions
    transitions = world.get_important_transitions()
    object_trans = transitions['object_transitions']
    
    print(f"Found {len(object_trans)} object transitions")
    
    # Test picking up an object
    print("\nTest 1: Picking up an object")
    # Find a table with an object
    table_with_object = None
    for table_pos in world.table_positions.keys():
        if world.grid[table_pos] >= 2:  # Has an object
            table_with_object = table_pos
            break
    
    if table_with_object:
        print(f"  Testing with table at {table_with_object} with object {world.grid[table_with_object]}")
        
        # Set agent to table position
        world.agent_pos = table_with_object
        world.agent_inventory = None
        
        # Execute toggle action
        world.step(4)  # TOGGLE action
        
        # Check if object was picked up
        if world.agent_inventory is not None and world.grid[table_with_object] == 0:
            print(f"  ✓ SUCCESS: Object {world.agent_inventory} picked up from table")
        else:
            print(f"  ✗ FAILURE: Object not picked up. Inventory: {world.agent_inventory}, Table: {world.grid[table_with_object]}")
    else:
        print("  ✗ SKIPPED: No table with object found")
    
    # Test putting down an object
    print("\nTest 2: Putting down an object")
    # Find an empty table
    empty_table = None
    for table_pos in world.table_positions.keys():
        if world.grid[table_pos] == 0:  # Empty table
            empty_table = table_pos
            break
    
    if empty_table and world.agent_inventory is not None:
        print(f"  Testing with empty table at {empty_table}, agent has object {world.agent_inventory}")
        
        # Set agent to empty table position
        world.agent_pos = empty_table
        
        # Execute toggle action
        world.step(4)  # TOGGLE action
        
        # Check if object was put down
        if world.agent_inventory is None and world.grid[empty_table] >= 2:
            print(f"  ✓ SUCCESS: Object {world.grid[empty_table]} put down on table")
        else:
            print(f"  ✗ FAILURE: Object not put down. Inventory: {world.agent_inventory}, Table: {world.grid[empty_table]}")
    else:
        print("  ✗ SKIPPED: No empty table found or agent has no object")


def test_invalid_actions():
    """Test that invalid actions don't change the agent's state"""
    print("\n=== Testing Invalid Actions ===")
    
    world = GridWorld(num_rooms=4, room_size=3)
    
    # Test moving into a wall
    print("Test 1: Moving into a wall")
    # Find a wall position next to the agent
    world.agent_pos = (0, 1)  # Top row
    start_pos = world.agent_pos
    print(f"  Start position: {start_pos}")
    
    # Try to move up (should fail)
    world.step(0)  # UP
    
    if world.agent_pos == start_pos:
        print("  ✓ SUCCESS: Agent did not move into wall")
    else:
        print(f"  ✗ FAILURE: Agent moved to {world.agent_pos}")
    
    # Test picking up object when not at table
    print("\nTest 2: Picking up object when not at table")
    # Find a non-table position that's not a wall
    # Let's use a position that's definitely not a table (not in table_positions)
    non_table_pos = None
    for x in range(world.grid_size):
        for y in range(world.grid_size):
            if (x, y) not in world.table_positions and world.grid[x, y] != 1:  # Not a table and not a wall
                non_table_pos = (x, y)
                break
        if non_table_pos:
            break
    
    if non_table_pos:
        world.agent_pos = non_table_pos
        world.agent_inventory = None
        start_inventory = world.agent_inventory
        print(f"  Start position: {world.agent_pos}, Inventory: {start_inventory}")
        
        # Try to pick up (should fail)
        world.step(4)  # TOGGLE
        
        if world.agent_inventory == start_inventory:
            print("  ✓ SUCCESS: Inventory unchanged when not at table")
        else:
            print(f"  ✗ FAILURE: Inventory changed to {world.agent_inventory}")
    else:
        print("  ✗ SKIPPED: Could not find a non-table position")


if __name__ == "__main__":
    # Run all tests
    test_door_transitions()
    test_object_transitions()
    test_invalid_actions()
    
    # Show final state of the test world
    print("\n=== Final Test World State ===")
    world = GridWorld(num_rooms=4, room_size=3)
    world.render()
    
    # Show some precomputed transitions
    transitions = world.get_important_transitions()
    print(f"\nSample Door Transitions:")
    for trans in transitions['door_transitions'][:3]:
        print(f"  {trans}")
    
    print(f"\nSample Object Transitions:")
    for trans in transitions['object_transitions'][:3]:
        print(f"  {trans}")