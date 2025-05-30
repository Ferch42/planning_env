import gym
from gym import spaces
import numpy as np
import time
import pickle
from misc import generate_combinations
from tqdm import tqdm

q = {}
action_count = {}

possible_prob_dist = [[2.5,2.5,2.5,2.5], [7, 1, 1, 1], [1, 7, 1, 1], [1, 1, 7, 1], [1, 1, 1, 7]] + [x for x in generate_combinations(10,4) if 4 in x and 3 in x and 2 in x and 1 in x]

def Q(state):

    global q
    if state not in q:
        q[state] = np.zeros(4)

    return q[state]

def reset_Q():

    global q    
    q = {}

def ACTION_COUNT(state):

    global action_count
    if state not in action_count:
        action_count[state] = np.zeros(4)
    
    return action_count[state]


class GridWorldEnv(gym.Env):
    """Grid world with dynamic goals, visited corners, and fixed-position objects."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Fixed possible object positions
    FIXED_POSITIONS = [
        (7,7), (7,10), (7,13),
        (10, 7), (10,13),
        (13,7), (13,10), (13,13)
    ]

    def __init__(self, render_mode=None, 
                 goals_prob_distribution=[0.25]*4,
                 object_prob_dist=[0.25]*4,
                 num_objects= 4 ):
        super().__init__()
        self.size = 21
        self.agent_start_pos = (self.size//2, self.size//2)
        self.corners = [(0, 0), (self.size-1, 0), 
                       (0, self.size-1), (self.size-1, self.size-1)]
        
        # Validate inputs
        self._validate_parameters(goals_prob_distribution, object_prob_dist, num_objects)
        
        self.goals_prob_distribution = goals_prob_distribution
        self.object_prob_dist = object_prob_dist
        self.num_objects = num_objects
        self.objects = {}  # {(x,y): 'A/B/C/D'}
        self.collected_objects = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.goal_pos = None
        self.visited_corners = np.zeros(4, dtype=np.int32)
        self.agent_pos = self.agent_start_pos

        # Observation space remains unchanged
        self.observation_space = spaces.MultiDiscrete(
            [self.size, self.size] + [2]*4
        )
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode

    def _validate_parameters(self, goals_dist, obj_dist, num_objects):
        if len(goals_dist) != 4 or not np.isclose(sum(goals_dist), 1.0):
            raise ValueError("Invalid goal distribution")
        if len(obj_dist) != 4 or not np.isclose(sum(obj_dist), 1.0):
            raise ValueError("Invalid object distribution - need 4 probabilities for A,B,C,D")
        if num_objects < 0 or num_objects > len(self.FIXED_POSITIONS):
            raise ValueError(f"Number of objects must be between 0 and {len(self.FIXED_POSITIONS)}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.goal_pos = self.corners[self.np_random.choice(4, p=self.goals_prob_distribution)]
        self.agent_pos = self.agent_start_pos
        self.visited_corners[:] = 0
        self.collected_objects = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self._spawn_objects()
        return self._get_obs(), self._get_info()

    def _spawn_objects(self):
        self.objects = {}
        # Randomly select positions without replacement
        selected_positions = self.np_random.choice(
            len(self.FIXED_POSITIONS),
            size=self.num_objects,
            replace=False
        )
        
        for idx in selected_positions:
            pos = self.FIXED_POSITIONS[idx]
            # Don't spawn on agent start or goal
            if pos == self.agent_start_pos or pos == self.goal_pos:
                continue
                
            obj_type = self.np_random.choice(['A', 'B', 'C', 'D'], p=self.object_prob_dist)
            self.objects[pos] = obj_type

    def step(self, action):
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = deltas[action]
        new_pos = (
            np.clip(self.agent_pos[0] + dx, 0, self.size-1),
            np.clip(self.agent_pos[1] + dy, 0, self.size-1)
        )
        self.agent_pos = new_pos

        # Check for object collection
        collected_type = None
        if self.agent_pos in self.objects:
            collected_type = self.objects.pop(self.agent_pos)
            self.collected_objects[collected_type] += 1

        # Update visited corners
        if self.agent_pos in self.corners:
            idx = self.corners.index(self.agent_pos)
            self.visited_corners[idx] = 1

        terminated = self.agent_pos == self.goal_pos
        reward = 1.0 if terminated else 0.0
        truncated = False

        info = self._get_info()
        if collected_type:
            info['collected_this_step'] = collected_type

        #if self.render_mode == "human":
        #    self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([
            *self.agent_pos,
            *self.visited_corners
        ], dtype=np.int32)

    def _get_info(self):
        return {
            'goal_pos': self.goal_pos,
            'collected_objects': self.collected_objects.copy(),
            'objects_remaining': len(self.objects),
            'object_positions': list(self.objects.keys())
        }

    def render(self):
        if self.render_mode == "human":
            grid = []
            for y in range(self.size):
                row = []
                for x in range(self.size):
                    pos = (x, y)
                    if pos == self.agent_pos:
                        row.append("A")
                    elif pos == self.goal_pos:
                        row.append("G")
                    elif pos in self.objects:
                        row.append(self.objects[pos].lower())
                    else:
                        row.append(".")
                grid.append(" ".join(row))
            print("\n".join(grid))
            print(f"Visited corners: {self.visited_corners}")
            print(f"Objects remaining: {len(self.objects)}")
            print(f"GOAL: {self.goal_pos}")
        
        elif self.render_mode == "rgb_array":
            rgb_array = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
            # Goal (green)
            rgb_array[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]
            # Agent (red)
            ax, ay = self.agent_pos
            rgb_array[ay, ax] = [255, 0, 0]
            # Objects (A: blue, B: yellow, C: purple, D: cyan)
            for (x, y), obj in self.objects.items():
                if obj == 'A':
                    rgb_array[y, x] = [0, 0, 255]
                elif obj == 'B':
                    rgb_array[y, x] = [255, 255, 0]
                elif obj == 'C':
                    rgb_array[y, x] = [128, 0, 128]
                elif obj == 'D':
                    rgb_array[y, x] = [0, 255, 255]
            return rgb_array

    def close(self):
        pass

    def transform_obs(self,obs):
        
        n_cells = self.size ** 2
        return obs[0] + obs[1] * self.size + n_cells * (obs[2]*(2**0) + obs[3]*(2**1) + obs[4]*(2**2) + obs[5]*(2**3))

# Example usage
if __name__ == "__main__":

    # Hyperparameter configuration
    ALPHA = 0.1
    GAMMA = 0.99
    
    env = GridWorldEnv(
        goals_prob_distribution= [0.25,0.25,0.25,0.25],
        object_prob_dist=[0.4, 0.3, 0.2, 0.1],  # 40% A, 30% B, 20% C, 10% D
        num_objects= 4,
        render_mode="human"
    )
        
    timestep_list = []
    possible_states = set()

    for e in tqdm(range(10_000)):

        s, info = env.reset()
        done = False
        s = env.transform_obs(s)
        t = 0
        
        while not done:
            
            possible_states.add(s)
            action = env.action_space.sample()
                
            ss, reward, terminated, _, info = env.step(action)
            t += 1
            
            done = terminated 
            ss = env.transform_obs(ss)
            for ps in possible_states:
                r = int(ss == ps) 
                extended_s = (s, ps)
                extended_ss = (ss, ps)
                Q(extended_s)[action] = Q(extended_s)[action] + ALPHA* (r + GAMMA * Q(extended_ss).max() - Q(extended_s)[action])

            s = ss
            if done:
                timestep_list.append(t)
                #time.sleep(1)
                 
            


    with open('q.pkl', 'wb+') as f:
        pickle.dump(q, f)

    