import numpy as np
import cv2

import time
import gym
import matplotlib.pyplot as plt
from matplotlib import animation


def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
    anim.save('breakout_result.gif', fps=30)


ENV = 'CartPole-v1'
NUM_DIGITIZED = 6
GAMMA = 0.99  # decrease rate
ETA = 0.7  # learning rate
MAX_STEPS = 1000  # steps for 1 episode
NUM_EPISODES = 300  # number of episodes


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next)

    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1: -1]  # num of bins needs num+1 value


class Brain:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # the number of CartPole actions

        self.q_table = np.random.uniform(low=0, high=1, size=(
            NUM_DIGITIZED ** num_states, num_actions))

    @staticmethod
    def digitize_state(observation):
        cart_pos, cart_v, pole_angle, pole_v = observation

        digitized = [
            np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIGITIZED)),
            np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIGITIZED)),
            np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIGITIZED)),  # angle represent by radian
            np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIGITIZED))
        ]

        return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
                                      ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)

        return action


class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            for step in range(MAX_STEPS):
                self.env.render()

                if is_episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(observation, episode)  # not step

                observation_next, _, done, _ = self.env.step(action)

                if done:
                    if step < 100:
                        reward = -1
                        complete_episodes = 0
                    elif 100 <= step < 200:
                        reward = 0
                        complete_episodes = 0
                    elif 200 <= step < 300:
                        reward = 1
                        complete_episodes = 0
                    else:
                        reward = 2
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.update_Q_function(observation, action, reward, observation_next)

                observation = observation_next

                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

            if is_episode_final is True:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('succeeded for 10 times')
                is_episode_final = True


if __name__ == "__main__":
    env = Environment()
    env.run()
