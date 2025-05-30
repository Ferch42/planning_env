import pickle
from env import GridWorldEnv
import numpy as np
import time


with open('policies.pkl', 'rb') as f:
    data = pickle.load(f)

#print(data.keys())

P_MATRIX = np.array([[0.60,0.15,0.15,0.10],
                     [0.15,0.60,0.10,0.15],
                     [0.15,0.10,0.60,0.15],
                     [0.10,0.15,0.15,0.60]])

P_MATRIX_2 = np.array([[0.40,0.25,0.25,0.10],
                     [0.25,0.40,0.10,0.25],
                     [0.25,0.10,0.40,0.25],
                     [0.10,0.25,0.25,0.40]])

P_MATRIX_1 = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

P_MATRIX = P_MATRIX_1

OBJ_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def bayesian_update(prob_apriori, likelihood):

    p = prob_apriori * likelihood

    return p/p.sum()


PROB_DISTR_ORDER ={
 (2.5, 2.5, 2.5, 2.5) : [3, 2, 1, 0], 
 (7, 1, 1, 1): [0, 1, 2, 3], 
 (1, 7, 1, 1): [1, 0, 3, 2], 
 (1, 1, 7, 1): [2, 1, 0, 3], 
 (1, 1, 1, 7): [1, 3, 2, 0],
 (1, 2, 3, 4): [3, 2, 1, 0],
 (1, 2, 4, 3): [3, 2, 0, 1],
 (1, 3, 2, 4): [3, 1, 2, 0],
 (1, 3, 4, 2): [3, 1, 0, 2],
 (1, 4, 2, 3): [3, 0, 2, 1],
 (1, 4, 3, 2): [3, 0, 1, 2],
 (2, 1, 3, 4): [2, 3, 1, 0],
 (2, 1, 4, 3): [2, 3, 0, 1],
 (2, 3, 1, 4): [2, 1, 3, 0],
 (2, 3, 4, 1): [2, 1, 0, 3],
 (2, 4, 1, 3): [2, 0, 3, 1],
 (2, 4, 3, 1): [2, 0, 1, 3],
 (3, 1, 2, 4): [1, 3, 2, 0],
 (3, 1, 4, 2): [1, 3, 0, 2],
 (3, 2, 1, 4): [1, 2, 3, 0],
 (3, 2, 4, 1): [1, 2, 0, 3],
 (3, 4, 1, 2): [1, 0, 3, 2],
 (3, 4, 2, 1): [1, 0, 2, 3],
 (4, 1, 2, 3): [0, 3, 2, 1],
 (4, 1, 3, 2): [0, 3, 1, 2],
 (4, 2, 1, 3): [0, 2, 3, 1],
 (4, 2, 3, 1): [0, 2, 1, 3],
 (4, 3, 1, 2): [0, 1, 3, 2],
 (4, 3, 2, 1): [0, 1, 2, 3]
}


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


mean_list = []
for i in range(1,9):

    timestep_list = []

    for e in range(10000):
        #print(e)
        goal = np.random.choice(range(4))
        goal_distr = np.zeros(4)
        goal_distr[goal] = 1 
        evidence_distribution = P_MATRIX[goal]

        env = GridWorldEnv(
            goals_prob_distribution= goal_distr,
            object_prob_dist= evidence_distribution ,  # 40% A, 30% B, 20% C, 10% D
            num_objects= i,
            render_mode="human"
        )
        s, info = env.reset()
        
        done = False
        t = 0
        p = np.array([0.25,0.25,0.25,0.25])

        policy = PROB_DISTR_ORDER[(2.5, 2.5, 2.5, 2.5)]

        while not done:
            
            if t<30:
                action = np.random.choice(range(4))
            else:
                action = corner_visit_policy(s, policy, env)    
            

            ss, reward, terminated, _, info = env.step(action)
            t+=1
            if 'collected_this_step' in info:
                #print(info['collected_this_step'])
                p = bayesian_update(p, P_MATRIX[:, OBJ_IDX[info['collected_this_step']]])
                #print(p)
                possible_distributions = [(x, np.array(x)) for x  in PROB_DISTR_ORDER]
                best_distribution = sorted(possible_distributions, key = lambda x: np.sqrt(np.sum((x[1] - p *10)**2)))[0][0]
                #print(p, best_distribution)
                #print(goal_distr)
                policy = PROB_DISTR_ORDER[best_distribution]
            

            s = ss
            done = terminated

        timestep_list.append(t)
    #print(f"Hyper parameter : {prob_dist}")
    print(f"Mean : {np.mean(timestep_list)}")
    mean_list.append(np.mean(timestep_list))

print(mean_list)