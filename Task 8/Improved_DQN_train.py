"""
Name: Nima Jamshidi
Professor Leilani Gilpin
CMPM-118
Task 7 (AIEA)
May 29th, 2026
"""

import time
import torch
import random
import numpy as np
import gymnasium as gym
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Parameters:
TOTAL_STEPS = 300_000
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 200_000
TRAIN_START = 5_000
LEARN_INCREMENT = 4
TARGET_UPDATE = 2_000
EPS_START = 1.0
EPS_END = 0.10
EPS_DELAY_STEPS = 250_000
SEED = 1
REPEAT_ACTION = 4

# Helper Functions:

"""
Pre_process_frame() takes a raw frame and converts it to the grayscale, resizes 
it to 84x84 and returns the smaller image
"""
def pre_process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_resize = cv2.resize(gray_frame, (84, 84))
    return frame_resize

"""
FrameStacker() is a class that is supposed to store the last 4 frames 
"""
class FrameStacker:
    def __init__(self, stack_size=4):
        self.frames = deque(maxlen=stack_size)
    
    """
    reset() clears the frames that were remaining from the previous episode
    """
    def reset(self, first_frame):
        self.frames.clear()
        processed = pre_process_frame(first_frame)

        for _ in range(self.frames.maxlen):
            self.frames.append(processed)
        
        return np.stack(self.frames)
    
    """
    step() preprocesses the new frame, adds it to the buffer and return the stacked state
    """
    def step(self, frame):
        self.frames.append(pre_process_frame(frame))
        return np.stack(self.frames)

"""
esp_get() linearly decrease the esp for esp greedy action selection
"""    
def esp_get(step):
    if step >= EPS_DELAY_STEPS:
        return EPS_END
        
    x = step / EPS_DELAY_STEPS
    eps = EPS_START + x * (EPS_END - EPS_START)
    return eps
        
"""
BufferReplay() stores the actions. It takes the previous actions and samples them
randomly
"""
class BufferReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    """
    add() stores a single transistion inside the replay memory
    """
    def add(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))

    """
    sample() selects a minibatch of actions from the replay buffer and random sampling 
    prevents this by learning from constent observation
    """
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d))
    
    def __len__(self):
        return len(self.memory)

"""
DQN() is a helper function that the agent to make the decisions. Ot takes the current game
state as an input and predicts if the actions were good or not. Then the agent will choose 
the highest value 
"""
class DQN(nn.Module):
    def __init__(self, num_action):
        super().__init__()
        self.cnn_layer = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                       nn.ReLU(),
                                       nn.Flatten())
        
        with torch.no_grad():
            sample_in = torch.zeros(1, 4, 84, 84)
            flat_size = self.cnn_layer(sample_in).shape[1]

        self.out_layer = nn.Sequential(nn.Linear(flat_size, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, num_action))
        
    def forward(self, x):
        x = self.cnn_layer(x)
        return self.out_layer(x)
    
"""
DQN_train() combines everything to start training, it builds the neural networks to create
the replay buffer and logging system
"""
class DQN_train:
    def __init__(self):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        self.device = torch.device("cpu")
        self.env = gym.make("CarRacing-v3", continuous=False, domain_randomize=False)
        obs, _ = self.env.reset(seed=SEED)

        self.num_action = self.env.action_space.n
        self.frame_stacker = FrameStacker()
        self.state = self.frame_stacker.reset(obs)

        self.q_network = DQN(self.num_action).to(self.device)
        self.target_network = DQN(self.num_action).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = BufferReplay(BUFFER_SIZE)

        self.step_count = 0
        self.episode = 0
        self.eps_reward = 0
        self.eps_length = 0
        self.last_loss = 0.0

        run_name = time.strftime("dqn_nima_%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"runs/{run_name}")

    """
    choose_action() decides what action needs to be taken by the agent. It also has a random
    action to explore new possibilities so it can learn new things
    """
    def choose_action(self):
        esp = esp_get(self.step_count)

        if random.random() < esp:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
            state_tensor /= 255.0

            with torch.no_grad():
                q_val = self.q_network(state_tensor)
                action = int(q_val.argmax(dim=1).item())

        return action, esp
    
    """
    rollout_step() does one step in the enviroment and makes the agent choose an action
    rewards it and new state stores the experiement in the memory
    """
    def rollout_step(self):
        self.step_count += 1

        action, esp = self.choose_action()

        total_reward = 0
        done = False
        obs_next = None

        for _ in range(REPEAT_ACTION):
            obs_next, reward, term, trunc, _ = self.env.step(int(action))
            total_reward += reward
            done = term or trunc

            if done:
                break

        clipped_reward = np.clip(total_reward, -1.0, 1.0)

        next_state = self.frame_stacker.step(obs_next)

        self.replay_buffer.add(self.state, int(action), clipped_reward, next_state, done)
        self.state = next_state
        self.eps_reward += total_reward
        self.eps_length += 1

        self.writer.add_scalar("train/epsilon", esp, self.step_count)

        if done:
            self.episode += 1

            print(
                f"Episode {self.episode}, "
                f"Reward: {self.eps_reward:.2f}, "
                f"Loss: {self.last_loss:.4f}"
            )

            self.writer.add_scalar("episode/reward", self.eps_reward, self.episode)
            self.writer.add_scalar("episode/length", self.eps_length, self.episode)

            obs, _ = self.env.reset()
            self.state = self.frame_stacker.reset(obs)

            self.eps_reward = 0
            self.eps_length = 0

    """
    train_step() trains the actual DQN. It takes the past experiements from the replay buffer 
    and then calculates the Q-values to see if the weights to improve it's future predictions
    """
    def train_step(self):
        if self.step_count < TRAIN_START:
            return

        if self.step_count % LEARN_INCREMENT != 0:
            return

        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.from_numpy(states).float().to(self.device) / 255.0
        next_states = torch.from_numpy(next_states).float().to(self.device) / 255.0
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).float().to(self.device)

        current_q = self.q_network(states)
        chosen_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_target = self.target_network(next_states)
            max_next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * (1 - dones) * max_next_q

        loss = nn.functional.smooth_l1_loss(chosen_q, target_q)
        self.last_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.writer.add_scalar("train/loss", loss.item(), self.step_count)
    
    def update_target(self):
        if self.step_count % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.writer.add_scalar("train/target_update", 1, self.step_count)


    """
    learn() repeatedly collects experiences and trains DQN and updates the target until the 
    total steps is reached
    """
    def learn(self):
        while self.step_count < TOTAL_STEPS:
            self.rollout_step()
            self.train_step()
            self.update_target()

        self.env.close()
        self.writer.close()
        torch.save(self.q_network.state_dict(), "dqn_model.pt")

def main():
    trainer = DQN_train()
    trainer.learn()

if __name__ == "__main__":
    main()