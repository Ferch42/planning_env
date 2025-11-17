from graphviz import Digraph

# Create final graph with Lapis→Gold trade and Meat
dot1 = Digraph(comment='Minecraft Final System')
dot1.attr(rankdir='TB', size='18,14')

# Define node styles
location_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightblue'}
object_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightyellow'}
tool_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgray'}
weapon_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'pink'}
final_goal_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgreen', 'peripheries': '2'}

# Location nodes
dot1.node('forest', 'Forest', **location_style)
dot1.node('mine', 'Mine', **location_style)
dot1.node('orchard', 'Apple Orchard', **location_style)
dot1.node('craft', 'Crafting Table', **location_style)
dot1.node('village', 'Village', **location_style)

# Basic Resource nodes
dot1.node('wood', 'Wood', **object_style)
dot1.node('sticks', 'Sticks', **object_style)
dot1.node('stone', 'Stone', **object_style)
dot1.node('iron', 'Iron', **object_style)
dot1.node('diamonds', 'Diamonds', **object_style)
dot1.node('string', 'String', **object_style)
dot1.node('gold', 'Gold', **object_style)
dot1.node('apple', 'Apple', **object_style)
dot1.node('lapis', 'Lapis Lazuli', **object_style)
dot1.node('meat', 'Meat', **object_style)  # REINTRODUCED

# Early Game Weapons
dot1.node('wood_sword', 'Wood Sword', **weapon_style)
dot1.node('stone_sword', 'Stone Sword', **weapon_style)

# Tool nodes
dot1.node('wood_pick', 'Wood Pickaxe', **tool_style)
dot1.node('iron_pick', 'Iron Pickaxe', **tool_style)

# Final Goals
dot1.node('diamond_sword', 'Diamond Sword', **final_goal_style)
dot1.node('golden_apple', 'Golden Apple', **final_goal_style)
dot1.node('enchanted_bow', 'Enchanted Bow', **final_goal_style)

# Define colors
GREEN = 'green'
BLUE = 'blue'
ORANGE = 'orange'
RED = 'red'

# Gathering actions (green)
dot1.edge('forest', 'wood', color=GREEN)
dot1.edge('forest', 'sticks', color=GREEN)
dot1.edge('orchard', 'apple', color=GREEN)
dot1.edge('mine', 'stone', color=GREEN)
dot1.edge('mine', 'iron', color=GREEN)
dot1.edge('mine', 'diamonds', color=GREEN)
dot1.edge('mine', 'string', color=GREEN)
dot1.edge('mine', 'gold', color=GREEN)
dot1.edge('mine', 'lapis', color=GREEN)

# Early Game Weapons (blue)
dot1.edge('wood', 'craft', label='Craft', color=BLUE)
dot1.edge('craft', 'wood_sword', color=BLUE)
dot1.edge('stone', 'craft', label='Craft', color=BLUE)
dot1.edge('craft', 'stone_sword', color=BLUE)

# Hunting with early weapons (red)
dot1.edge('wood_sword', 'forest', label='Hunt', color=RED)
dot1.edge('stone_sword', 'forest', label='Hunt', color=RED)
dot1.edge('forest', 'meat', color=GREEN)

# Tool crafting (blue)
dot1.edge('wood', 'craft', label='+ Sticks', color=BLUE)
dot1.edge('sticks', 'craft', color=BLUE)
dot1.edge('craft', 'wood_pick', color=BLUE)

dot1.edge('iron', 'craft', label='+ Sticks', color=BLUE)
dot1.edge('sticks', 'craft', color=BLUE)
dot1.edge('craft', 'iron_pick', color=BLUE)

# Tool usage (red)
dot1.edge('wood_pick', 'mine', label='Mine Iron', color=RED)
dot1.edge('iron_pick', 'mine', label='Mine Diamonds/Gold/Lapis', color=RED)

# Final Goals crafting (blue) - 1 copy each
dot1.edge('diamonds', 'craft', label='+ Sticks', color=BLUE)
dot1.edge('sticks', 'craft', color=BLUE)
dot1.edge('craft', 'diamond_sword', color=BLUE)

dot1.edge('apple', 'craft', label='+ Gold', color=BLUE)
dot1.edge('gold', 'craft', color=BLUE)
dot1.edge('craft', 'golden_apple', color=BLUE)

dot1.edge('string', 'craft', label='+ Lapis', color=BLUE)
dot1.edge('lapis', 'craft', color=BLUE)
dot1.edge('craft', 'enchanted_bow', color=BLUE)

# 1:1 Trading actions (orange)
dot1.edge('wood', 'village', label='Trade wood\nfor Iron Pick', color=ORANGE)
dot1.edge('village', 'iron_pick', color=ORANGE)

dot1.edge('sticks', 'village', label='Trade sticks\nfor String', color=ORANGE)
dot1.edge('village', 'string', color=ORANGE)

dot1.edge('stone', 'village', label='Trade stone\nfor Lapis', color=ORANGE)
dot1.edge('village', 'lapis', color=ORANGE)

dot1.edge('lapis', 'village', label='Trade lapis\nfor Gold', color=ORANGE)  # CHANGED
dot1.edge('village', 'gold', color=ORANGE)

dot1.edge('meat', 'village', label='Trade meat\nfor Diamonds', color=ORANGE)  # REINTRODUCED
dot1.edge('village', 'diamonds', color=ORANGE)

dot1.render('minecraft_final_main', format='png', cleanup=True)
print("Final main graph generated as 'minecraft_final_main.png'")

# Create final goal paths graph
dot2 = Digraph(comment='Final Goal Paths')
dot2.attr(rankdir='TB', size='18,14')

# Early Game Weapons Section
with dot2.subgraph(name='cluster_early_weapons') as early:
    early.attr(label='Early Game (Required for Meat)', style='rounded', color='brown', fontsize='14')
    
    early.node('ew_forest', 'Forest', **location_style)
    early.node('ew_wood', 'Wood', **object_style)
    early.node('ew_stone', 'Stone', **object_style)
    early.node('ew_craft', 'Crafting Table', **location_style)
    
    early.node('ew_wood_sword', 'Wood Sword', **weapon_style)
    early.node('ew_stone_sword', 'Stone Sword', **weapon_style)
    
    early.edge('ew_forest', 'ew_wood', color=GREEN)
    early.edge('ew_forest', 'ew_stone', color=GREEN)
    early.edge('ew_wood', 'ew_craft', label='Craft Wood Sword', color=BLUE)
    early.edge('ew_craft', 'ew_wood_sword', color=BLUE)
    early.edge('ew_stone', 'ew_craft', label='Craft Stone Sword', color=BLUE)
    early.edge('ew_craft', 'ew_stone_sword', color=BLUE)

# Diamond Sword Path
with dot2.subgraph(name='cluster_sword') as sword:
    sword.attr(label='Diamond Sword', style='rounded', color='red', fontsize='16')
    
    sword.node('s_forest', 'Forest', **location_style)
    sword.node('s_mine', 'Mine', **location_style)
    sword.node('s_craft', 'Crafting Table', **location_style)
    sword.node('s_village', 'Village', **location_style)
    
    sword.node('s_wood', 'Wood', **object_style)
    sword.node('s_sticks', 'Sticks', **object_style)
    sword.node('s_iron', 'Iron', **object_style)
    sword.node('s_diamonds', 'Diamonds', **object_style)
    sword.node('s_meat', 'Meat', **object_style)
    
    sword.node('s_wood_pick', 'Wood Pickaxe', **tool_style)
    sword.node('s_iron_pick', 'Iron Pickaxe', **tool_style)
    sword.node('s_early_sword', 'Wood/Stone Sword', **weapon_style)
    sword.node('s_sword', 'Diamond Sword', **final_goal_style)
    
    # Connect to early weapons
    sword.edge('ew_wood_sword', 's_early_sword', style='dashed', color='purple')
    sword.edge('ew_stone_sword', 's_early_sword', style='dashed', color='purple')
    
    # Traditional Mining Path
    sword.edge('s_forest', 's_wood', color=GREEN)
    sword.edge('s_forest', 's_sticks', color=GREEN)
    sword.edge('s_wood', 's_craft', label='Craft Wood Pick', color=BLUE)
    sword.edge('s_sticks', 's_craft', color=BLUE)
    sword.edge('s_craft', 's_wood_pick', color=BLUE)
    
    sword.edge('s_wood_pick', 's_mine', label='Mine Iron', color=RED)
    sword.edge('s_mine', 's_iron', color=GREEN)
    sword.edge('s_iron', 's_craft', label='Craft Iron Pick', color=BLUE)
    sword.edge('s_craft', 's_iron_pick', color=BLUE)
    
    sword.edge('s_iron_pick', 's_mine', label='Mine Diamonds', color=RED)
    sword.edge('s_mine', 's_diamonds', color=GREEN)
    sword.edge('s_diamonds', 's_craft', label='Craft Sword', color=BLUE)
    sword.edge('s_craft', 's_sword', color=BLUE)
    
    # Alternative: Trade for tools
    sword.edge('s_wood', 's_village', label='Trade wood\nfor Iron Pick', color=ORANGE)
    sword.edge('s_village', 's_iron_pick', color=ORANGE)
    sword.edge('s_iron_pick', 's_mine', label='Mine Diamonds', color=RED)
    sword.edge('s_mine', 's_diamonds', color=GREEN)
    
    # Alternative: Meat Trading
    sword.edge('s_early_sword', 's_forest', label='Hunt Animals', color=RED)
    sword.edge('s_forest', 's_meat', color=GREEN)
    sword.edge('s_meat', 's_village', label='Trade meat\nfor Diamonds', color=ORANGE)
    sword.edge('s_village', 's_diamonds', color=ORANGE)

# Golden Apple Path
with dot2.subgraph(name='cluster_apple') as apple:
    apple.attr(label='Golden Apple', style='rounded', color='gold', fontsize='16')
    
    apple.node('a_orchard', 'Apple Orchard', **location_style)
    apple.node('a_forest', 'Forest', **location_style)
    apple.node('a_mine', 'Mine', **location_style)
    apple.node('a_craft', 'Crafting Table', **location_style)
    apple.node('a_village', 'Village', **location_style)
    
    apple.node('a_wood', 'Wood', **object_style)
    apple.node('a_sticks', 'Sticks', **object_style)
    apple.node('a_stone', 'Stone', **object_style)
    apple.node('a_iron', 'Iron', **object_style)
    apple.node('a_gold', 'Gold', **object_style)
    apple.node('a_apple', 'Apple', **object_style)
    apple.node('a_lapis', 'Lapis Lazuli', **object_style)
    
    apple.node('a_wood_pick', 'Wood Pickaxe', **tool_style)
    apple.node('a_iron_pick', 'Iron Pickaxe', **tool_style)
    apple.node('a_golden_apple', 'Golden Apple', **final_goal_style)
    
    # Get Apple
    apple.edge('a_orchard', 'a_apple', color=GREEN)
    
    # Path A: Mining Gold
    apple.edge('a_forest', 'a_wood', color=GREEN)
    apple.edge('a_forest', 'a_sticks', color=GREEN)
    apple.edge('a_wood', 'a_craft', label='Craft Wood Pick', color=BLUE)
    apple.edge('a_sticks', 'a_craft', color=BLUE)
    apple.edge('a_craft', 'a_wood_pick', color=BLUE)
    
    apple.edge('a_wood_pick', 'a_mine', label='Mine Iron', color=RED)
    apple.edge('a_mine', 'a_iron', color=GREEN)
    apple.edge('a_iron', 'a_craft', label='Craft Iron Pick', color=BLUE)
    apple.edge('a_craft', 'a_iron_pick', color=BLUE)
    
    apple.edge('a_iron_pick', 'a_mine', label='Mine Gold', color=RED)
    apple.edge('a_mine', 'a_gold', color=GREEN)
    
    # Path B: Tool Trading + Mining
    apple.edge('a_wood', 'a_village', label='Trade wood\nfor Iron Pick', color=ORANGE)
    apple.edge('a_village', 'a_iron_pick', color=ORANGE)
    apple.edge('a_iron_pick', 'a_mine', label='Mine Gold', color=RED)
    apple.edge('a_mine', 'a_gold', color=GREEN)
    
    # Path C: Stone Trading for Lapis → Gold (FIXED: removed duplicate trade)
    apple.edge('a_mine', 'a_stone', color=GREEN)
    apple.edge('a_stone', 'a_village', label='Trade stone\nfor Lapis', color=ORANGE)
    apple.edge('a_village', 'a_lapis', color=ORANGE)
    # Only one lapis→gold trade edge now
    apple.edge('a_lapis', 'a_village', label='Trade lapis\nfor Gold', color=ORANGE)
    apple.edge('a_village', 'a_gold', color=ORANGE)
    
    # Path D: Mining Lapis + Trading
    apple.edge('a_forest', 'a_wood', color=GREEN)
    apple.edge('a_forest', 'a_sticks', color=GREEN)
    apple.edge('a_wood', 'a_craft', label='Craft Wood Pick', color=BLUE)
    apple.edge('a_sticks', 'a_craft', color=BLUE)
    apple.edge('a_craft', 'a_wood_pick', color=BLUE)
    
    apple.edge('a_wood_pick', 'a_mine', label='Mine Iron', color=RED)
    apple.edge('a_mine', 'a_iron', color=GREEN)
    apple.edge('a_iron', 'a_craft', label='Craft Iron Pick', color=BLUE)
    apple.edge('a_craft', 'a_iron_pick', color=BLUE)
    
    apple.edge('a_iron_pick', 'a_mine', label='Mine Lapis', color=RED)
    apple.edge('a_mine', 'a_lapis', color=GREEN)
    # This path also uses the same lapis→gold trade above
    
    # Craft Golden Apple
    apple.edge('a_apple', 'a_craft', label='Craft Golden Apple', color=BLUE)
    apple.edge('a_gold', 'a_craft', color=BLUE)
    apple.edge('a_craft', 'a_golden_apple', color=BLUE)

# Enchanted Bow Path
with dot2.subgraph(name='cluster_bow') as bow:
    bow.attr(label='Enchanted Bow', style='rounded', color='purple', fontsize='16')
    
    bow.node('b_forest', 'Forest', **location_style)
    bow.node('b_mine', 'Mine', **location_style)
    bow.node('b_craft', 'Crafting Table', **location_style)
    bow.node('b_village', 'Village', **location_style)
    
    bow.node('b_wood', 'Wood', **object_style)
    bow.node('b_sticks', 'Sticks', **object_style)
    bow.node('b_stone', 'Stone', **object_style)
    bow.node('b_iron', 'Iron', **object_style)  # FIXED: Added missing iron node
    bow.node('b_string', 'String', **object_style)
    bow.node('b_lapis', 'Lapis Lazuli', **object_style)
    
    bow.node('b_wood_pick', 'Wood Pickaxe', **tool_style)
    bow.node('b_iron_pick', 'Iron Pickaxe', **tool_style)
    bow.node('b_enchanted_bow', 'Enchanted Bow', **final_goal_style)
    
    # Path A: Traditional Mining
    bow.edge('b_forest', 'b_wood', color=GREEN)
    bow.edge('b_forest', 'b_sticks', color=GREEN)
    bow.edge('b_wood', 'b_craft', label='Craft Wood Pick', color=BLUE)
    bow.edge('b_sticks', 'b_craft', color=BLUE)
    bow.edge('b_craft', 'b_wood_pick', color=BLUE)
    
    bow.edge('b_wood_pick', 'b_mine', label='Mine Iron', color=RED)
    bow.edge('b_mine', 'b_iron', color=GREEN)
    bow.edge('b_iron', 'b_craft', label='Craft Iron Pick', color=BLUE)
    bow.edge('b_craft', 'b_iron_pick', color=BLUE)
    
    bow.edge('b_iron_pick', 'b_mine', label='Mine String & Lapis', color=RED)
    bow.edge('b_mine', 'b_string', color=GREEN)
    bow.edge('b_mine', 'b_lapis', color=GREEN)
    
    # Path B: Full Trading
    bow.edge('b_forest', 'b_sticks', color=GREEN)
    bow.edge('b_sticks', 'b_village', label='Trade sticks\nfor String', color=ORANGE)
    bow.edge('b_village', 'b_string', color=ORANGE)
    
    bow.edge('b_mine', 'b_stone', color=GREEN)
    bow.edge('b_stone', 'b_village', label='Trade stone\nfor Lapis', color=ORANGE)
    bow.edge('b_village', 'b_lapis', color=ORANGE)
    
    # Path C: Tool Trading + Mining
    bow.edge('b_wood', 'b_village', label='Trade wood\nfor Iron Pick', color=ORANGE)
    bow.edge('b_village', 'b_iron_pick', color=ORANGE)
    bow.edge('b_iron_pick', 'b_mine', label='Mine String & Lapis', color=RED)
    bow.edge('b_mine', 'b_string', color=GREEN)
    bow.edge('b_mine', 'b_lapis', color=GREEN)
    
    # Craft Enchanted Bow
    bow.edge('b_string', 'b_craft', label='Craft Enchanted Bow', color=BLUE)
    bow.edge('b_lapis', 'b_craft', color=BLUE)
    bow.edge('b_craft', 'b_enchanted_bow', color=BLUE)

dot2.render('minecraft_final_goals', format='png', cleanup=True)
print("Final goals graph generated as 'minecraft_final_goals.png'")