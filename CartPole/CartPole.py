import gym
import numpy as np
import cv2

ENV = 'CarRacing-v1'
ACTORS = 5
B, G, R = 0, 1, 2

environments = []
actor_done = [False] * ACTORS
rewards = [0] * ACTORS
images = [0] * ACTORS

scan_point = [[120, 95], [120, 95], [120, 95], [120, 96], [120, 96]]
dir = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]

for i in range(ACTORS):
    env = gym.make(ENV)
    observation = env.reset()
    environments.append(env)

H, W = environments[0].reset().shape[:2]
value = 0

t = 0
while t < 120:
    for i in range(ACTORS):
        if not actor_done[i]:
            env = environments[i]
            length = np.zeros(5, dtype=np.uint8)
            fin = [False] * 5
            observation, reward, done, _ = env.step((-0.2 + i / 10, 0.1, 0))
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)[:140]
            # scale_img = cv2.resize(observation, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
            # images[i] = scale_img

            for l in range(10, 41):
                for j in range(5):
                    if not fin[j]:
                        y0, x0 = scan_point[j]
                        y1, x1 = y0 + l * dir[j][0], x0 + l * dir[j][1]
                        color = observation[y1, x1]
                        # color = observation[scan_point[j][0], scan_point[j][0] + l * dir[j][0], scan_point[j][1] + l * dir[j][1]]
                        if color[G] > 160:
                            fin[j] = True
                            length[j] = l

            for j in range(5):
                y0, x0 = scan_point[j]
                if not fin[j]:
                    length[j] = 40
                y1, x1 = y0 + length[j] * dir[j][0], x0 + length[j] * dir[j][1]
                cv2.line(observation, (x0, y0), (x1, y1), (0, 0, 255), 1)

            images[i] = observation.copy()
            rewards[i] += reward
    cv2.imwrite(f"test/{t}.png", np.hstack(img for img in images))
    t += 1
