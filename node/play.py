#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import time
import cozmo
import rospy
import random
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import paho.mqtt.client as mqtt
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

steps_done = 0
current_angle = 0.0

def cozmo_program(robot: cozmo.robot.Robot):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 10000
    TARGET_UPDATE = 10

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    memory = ReplayMemory(10000)

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        print("gamma = {}".format(eps_threshold))
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.tensor([policy_net(state).argmax()])
        else:
            return torch.tensor([random.randrange(3)], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
        if non_final_mask.size()[0] < BATCH_SIZE: return
        non_final_mask = non_final_mask.reshape(BATCH_SIZE, 1)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        if non_final_next_states.size()[0] < BATCH_SIZE: return
        non_final_next_states = non_final_next_states.reshape(BATCH_SIZE, 1)
        state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, 1)
        action_batch = torch.cat(batch.action).reshape(BATCH_SIZE, 1)
        reward_batch = torch.cat(batch.reward).reshape(BATCH_SIZE, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device).reshape(BATCH_SIZE, 1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        print("... optimized")


    def on_connect(client, userdata, flags, respons_code):
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        global current_angle
        json_uft8 = str(msg.payload.decode("utf-8","ignore"))
        current_angle = float(json.loads(json_uft8)["rotation"][2])

    num_episodes = 50

    host = '127.0.0.1'
    port = 1883
    topic = '/cozmo/mocap'
    client = mqtt.Client(protocol=mqtt.MQTTv311)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port=port, keepalive=60)
    client.loop_start()
    rospy.Rate(1).sleep()

    for i_episode in range(1000):
        for t in range(100):
            if rospy.is_shutdown(): return
            print("loop = {}, {} -----".format(i_episode, t))
            # action
            angle = current_angle
            state = angle/math.pi
            print("state = {}".format(state))
            state = np.array([state], dtype=np.float32)
            state = torch.from_numpy(state).to(device)
            action = select_action(state)
            if action.item() == 0: cmd = (0, 0)
            elif action.item() == 1: cmd = (30, -30)
            elif action.item() == 2: cmd = (-30, 30)
            print("action = {}, cmd = {}".format(action.item(), cmd))
            robot.drive_wheels(cmd[0], cmd[1])
            rospy.Rate(1).sleep()
            # update state
            angle = current_angle
            reward = 1.0-abs(angle/math.pi)
            print("reward = {}".format(reward))
            reward = torch.tensor([reward], device=device)
            done = (t == (100-1))
            if not done:
                next_state = angle/math.pi
                next_state = np.array([next_state], dtype=np.float32)
                next_state = torch.from_numpy(next_state).to(device)
            else:
                next_state = None
            memory.push(state, action, next_state, reward)

        print("optimize model ...")
        optimize_model()

        reward_sum = 0.0
        if i_episode % TARGET_UPDATE == 0:
            print("update target and testing mode ...")
            robot.say_text("Test mode").wait_for_completed()
            target_net.load_state_dict(policy_net.state_dict())
            for t in range(30):
                if rospy.is_shutdown(): return
                angle = current_angle
                state = angle/math.pi
                state = np.array([state], dtype=np.float32)
                state = torch.from_numpy(state).to(device)
                action = policy_net(state).argmax().item()
                if action == 0: cmd = (0, 0)
                elif action == 1: cmd = (30, -30)
                elif action == 2: cmd = (-30, 30)
                robot.drive_wheels(cmd[0], cmd[1])
                rospy.Rate(1).sleep()
                angle = current_angle
                reward = 1.0-abs(angle/math.pi)
                reward_sum += reward
            print("### reward sum = {} ###".format(reward_sum))

        rospy.Rate(1).sleep()

    client.loop_stop()

if __name__ == "__main__":
    plt.ion()
    rospy.init_node("cozmo_play")
    cozmo.run_program(cozmo_program)
