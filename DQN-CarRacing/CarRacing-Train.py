import math
import random
import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from itertools import count

from ReplayMemory import ReplayMemory, Transition
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
plt.ion()

ENV = 'CarRacing-v1'
B, G, R = 0, 1, 2

env = gym.make(ENV)

scan_point = [[120, 95], [120, 95], [120, 95], [120, 96], [120, 96]]
directions = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]

H, W = env.reset().shape[:2]


BATCH_SIZE = 50
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY = 200
TARGET_UPDATE = 10
RENDER = True

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
criterion = nn.L1Loss()
memory = ReplayMemory(5000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1 * steps_done / EPSILON_DECAY)
    steps_done += 1
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state)
    else:
        return torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(0, 1.0), random.uniform(0, 1.0)]],
                            device=device, dtype=torch.float32)


episode_durations = []


def get_state(obs):
    obs = obs.copy()
    side_state = np.zeros(5, dtype=np.uint8)
    finish_scan = [False] * 5
    for l in range(10, 41):
        for j in range(5):
            if not finish_scan[j]:
                y0, x0 = scan_point[j]
                y1, x1 = y0 + l * directions[j][0], x0 + l * directions[j][1]
                color = obs[y1, x1]
                if color[G] > 160:
                    finish_scan[j] = True
                    side_state[j] = l

    if RENDER:
        for j in range(5):
            y0, x0 = scan_point[j]
            if not finish_scan[j]:
                side_state[j] = 40
            y1, x1 = y0 + side_state[j] * directions[j][0], x0 + side_state[j] * directions[j][1]
            cv2.line(obs, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.imshow("State", obs)
            cv2.waitKey(1)

    side_state = (side_state.astype(np.float32) - 10) / 40
    return side_state


def get_done(observation, reward, negative_reward_counter, time_frame_counter):
    negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0
    if negative_reward_counter > 20:
        return True
    if observation[129, 90, G] > 160 and observation[129, 101, G] > 160:
        return True


def plot_durations():
    plt.figure()
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float32)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.00001)
    plt.show()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=device,
                                  dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    next_state_batch = torch.cat(batch.next_state).to(device)

    state_action_values = policy_net(state_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # next_state_values = target_net(reward_batch)

    expect_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expect_state_action_values.unsqueeze(1))
    loss.backward()

    optimizer.step()


episodes = 3000
for episode in range(episodes):
    state = np.zeros(5)
    env.reset()
    action = (0, 0.1, 0)
    score = 0
    negative_reward_counter = 0
    time_frame_counter = 0
    for t in count():
        if RENDER:
            env.render()
        if t < 50:
            action = (0, 0, 0)
            last_observation, reward, _, _, = env.step(action)
            last_state = torch.tensor([get_state(last_observation)]).to(device)
            score += reward
        else:
            if t < 120:
                action = (0, 0.2, 0)
                current_observation, reward, game_done, _ = env.step(action)
                reward = torch.tensor([reward], device=device)
                score += reward
                action = torch.tensor([action], dtype=torch.float32, device=device)
            else:
                action = select_action(last_state)
                action = tuple(action.cpu().numpy()[0])
                # print(t, action)
                current_observation, reward, game_done, _ = env.step(action)
                if action[1] > 0.7 and action[2] < 0.1:
                    reward *= 2
                reward = torch.tensor([reward], device=device)
                action = torch.tensor([action], dtype=torch.float32, device=device)
            score += reward

            current_observation = cv2.cvtColor(current_observation, cv2.COLOR_BGR2RGB)[:140]
            state = torch.tensor([get_state(current_observation)], device=device)

            done = get_done(current_observation, reward, negative_reward_counter, time_frame_counter)
            if done or game_done:
                reward -= 10

            memory.push(last_state, action, reward, state, done)
            last_state = state

            optimize_model()
            time_frame_counter += 1

            if done:
                episode_durations.append(score)
                plot_durations()
                break
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

env.render()
env.close()
plt.ioff()
plt.show()
torch.save(policy_net, "test.pth")
dummy_input = torch.randn(1, 5, requires_grad=True).to(device)
torch.onnx.export(policy_net, dummy_input, "test.onnx", verbose=True, opset_version=13, input_names=["input"],
                  output_names=["output"])
