import pickle
from env import GridWorldEnv
import numpy as np
import time


def get_next_corner_target(state, order, env):
    """
    Determine the next corner to visit based on the specified order and visited flags
    
    Args:
        state: [agent_x, agent_y, visited0, visited1, visited2, visited3]
        order: List of corner indices specifying visitation order
        env: GridWorldEnv instance
    
    Returns:
        Tuple (x, y) of the next corner position to visit, or None if all corners visited
    """
    # Extract visited flags from state
    visited = state[2:6]
    
    # Find next unvisited corner in the specified order
    for corner_idx in order:
        if not visited[corner_idx]:
            return env.corners[corner_idx]
    
    # All corners have been visited
    return None


def get_action_towards_target(current_pos, target_pos, env):
    """
    Compute the best action to move from current position to target position
    
    Args:
        current_pos: Tuple (x, y) of current position
        target_pos: Tuple (x, y) of target position
        env: GridWorldEnv instance
    
    Returns:
        Integer action (0-3) that minimizes Manhattan distance to target
    """
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    best_action = 0
    best_distance = float('inf')
    
    for action in range(4):
        dx, dy = deltas[action]
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        
        # Clip to grid boundaries
        new_x = max(0, min(env.size-1, new_x))
        new_y = max(0, min(env.size-1, new_y))
        
        # Calculate Manhattan distance to target
        dist = abs(new_x - target_pos[0]) + abs(new_y - target_pos[1])
        
        # Update best action if this is better
        if dist < best_distance:
            best_distance = dist
            best_action = action
    
    return best_action


def corner_visit_policy(state, order, env):
    """
    Policy that visits corners in specified order
    
    Args:
        state: [agent_x, agent_y, visited0, visited1, visited2, visited3]
        order: List of corner indices specifying visitation order
        env: GridWorldEnv instance
    
    Returns:
        Integer action (0-3) to take
    """
    current_pos = (state[0], state[1])
    
    # Determine next corner target based on order and visited flags
    target_pos = get_next_corner_target(state, order, env)
    
    # If all corners visited, stay in place
    if target_pos is None:
        return 0  # Default to "up" action (but any action would work)
    
    # Get action toward the target corner
    return get_action_towards_target(current_pos, target_pos, env)


# Initialize environment
env = GridWorldEnv(render_mode="human", 
                   goals_prob_distribution=[0.25]*4,
                   object_prob_dist=[0.25]*4,
                   num_objects=4)

# Specify corner visitation order
corner_order = [2, 0, 1, 3]  # Visit corner2 first, then corner0, etc.

# Run an episode
obs, _ = env.reset()
terminated = False

while not terminated:
    # Get current state and select action
    action = corner_visit_policy(obs, corner_order, env)
    
    # Take action
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Render current state
    env.render()
    time.sleep(1)