from gridworld import GridWorld
import numpy as np


grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)

env = GridWorld(size=grid_size, start=start, goal=goal)
state = env.reset()

done = False
"""
while not done:
    action = np.random.choice(4)  # random policy
    state, reward, done = env.step(action)
    env.render()
    print("Reward:", reward, "Done:", done)
"""

policy = np.zeros((3, 3, 4))
# [up, down, left, right]
policy[0, 0, :] = [1, 0, 0, 1]
#print(policy)

#env.plot_policy((3, 3), policy)

env.render()
# down
state, reward, done = env.step(1)
state, reward, done = env.step(1)
state, reward, done = env.step(1)
state, reward, done = env.step(1)
# right
state, reward, done = env.step(3)
state, reward, done = env.step(3)
env.render()

print(state)
