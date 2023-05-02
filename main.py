import gym
import numpy as np
import random

# initialize environment and Q-table
env = gym.make('CliffWalking-v0')
qtable = np.zeros((48,4))

print("Q-table before training:")
print(qtable)

# training paramaters
episodes = 50
epsilon = 1
epsilon_decay = 0.02

# TRAINING
for _ in range(episodes):
    state, info = env.reset()
    done = False

    # keep running until episode terminated
    while not done:

        # exploration vs exploitation logic (increasingly more exploitative)
        randint = random.random()
        if epsilon > randint:
            # exploit
            action = env.action_space.sample()
        else:
            # explore
            action = np.argmax(qtable[state])

        # take action
        new_state, reward, done, transulated, info = env.step(action)

        # bellman equation (updates Q-value)
        qtable[state,action] = qtable[state,action] + (reward + np.max(qtable[new_state]) - qtable[state,action])

        # update state 
        state = new_state

    # decay epsilon value
    epsilon -= epsilon_decay

print()
print("Q-table after training:")
print(qtable)
print()

# EVALUATION
state, info = env.reset()
done = False
moves = 0
print("evaluation episode: ")
while not done:
    # always take exploitative move
    action = np.argmax(qtable[state])
    new_state, reward, done, transulated, info = env.step(action)
    # update state
    state = new_state

    # print move
    if action == 0: print("up")
    elif action == 1: print("right")
    elif action == 2: print("down")
    elif action == 3: print("left")
    moves += 1
    if new_state == 47: print(f"agent reached goal in: {moves} moves")
