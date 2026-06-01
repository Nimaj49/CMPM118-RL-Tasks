"""
/*
 * Name: Nima Jamshidi
 * Professor Leilani Gilpin
 * AIEA (Task 8) PPO Improvements
 * CMPM-118
*/
"""
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import gymnasium as gym


#parameters 
TOTAL_STEPS = 1_000_000
ROLLOUT_LENGTH = 2048
EPOCHS = 10
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-4
CLIP = 0.1
MAX_GRAD = 0.5
SEED = 1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MINIBATCH_SIZE = 128


# Helper Functions:

"""
pre_process is a helper function that is supposed to take a raw rgb CarRacing image and then 
converts to grayscale and resizes the image so that it can store the new value.
"""

def pre_process(obs_rgb):
    gray = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return gray.astype(np.uint8)


"""
reset_stack clears the old frames and preprocesses the first frame and copies it 
four times
"""

frame_buffer = deque(maxlen=4)
def reset_stack(observation):
    frame_buffer.clear()
    processed_frame = pre_process(observation)

    for x in range(4):
        frame_buffer.append(processed_frame)

    stacked_frames = np.stack(frame_buffer, axis=0)

    return stacked_frames

"""
update_stack preprocesses the new frame and adds it to the buffer frame so it can be returned after 
"""

def update_stack(observation):
    processed_frame = pre_process(observation)
    frame_buffer.append(processed_frame)
    stacked_frames = np.stack(frame_buffer, axis=0)

    return stacked_frames

"""
Note to self: This is the neural network of the PPO, it looks at the game and learns
information about the actual enviroment. The actor choose what actions the Algorithm should
do and the critic part estimates how good of an action it was. So if the action was good for the 
learning such as it predicts that a car is tilted left meaning that it is turning left, and let's say 
its destination is to the left, it would it reward it by giving it less negative reward output
"""
class ActorCriticNetwork(nn.Module):
    def __init__(self, number_of_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                nn.ReLU(),
                                nn.Flatten())
        
        with torch.no_grad():
            input = torch.zeros(1, 4, 84, 84)
            flat_size = self.feature_extractor(input).shape[1]

        self.policy_head = nn.Sequential(nn.Linear(flat_size, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, number_of_actions))
        
        self.val_head = nn.Sequential(nn.Linear(flat_size, 512),
                                        nn.ReLU(), 
                                        nn.Linear(512, 1))
        
    def forward(self, state):
        x = self.feature_extractor(state)
        logits = self.policy_head(x)
        state_val = self.val_head(x).squeeze(1)

        return logits, state_val    
    

"""
Calculate advantage estimates values for the PPO. It determines if the action was better or 
worst than expected. It helps the algorithm to decide which action it should be encourage to take
and what actions would not benefit much in the training.
"""

def gae_compute(reward, val, flags, final_val):
    advantage = []
    advantage_val = 0

    val += [final_val]

    for x in reversed(range(len(reward))):
        if flags[x]:
            state_next = 0

        else:
            state_next = 1

        td_error = (reward[x] + GAMMA * val[x + 1] * state_next - val[x])
        advantage_val = (td_error + GAMMA * LAMBDA * state_next *advantage_val)

        advantage.insert(0, advantage_val)


    returns = []

    for y, value in zip(advantage, val[:-1]):
        returns.append(y + value)


    return advantage, returns 

"""
Note to self: Main learning process. creates the enviroment and runs the agent to gather
information to train the neural network
"""

class PPO:
    def __init__(self):

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        self.device = torch.device("cpu")

        self.env = gym.make("CarRacing-v3", continuous=False, domain_randomize=False)
        first_obs, info = self.env.reset(seed=SEED)

        self.number_of_actions = self.env.action_space.n
        self.network = ActorCriticNetwork(self.number_of_actions).to(self.device)
        self.optimize = optim.Adam(self.network.parameters(), lr=LR)
        self.current_state = reset_stack(first_obs)

        self.step_count = 0
        self.ep_return = 0.0
        self.ep_length = 0
        self.ep_number = 0

        self.recent_returns = deque(maxlen=20)

        car_run = time.strftime("ppo_carracing_%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"runs/{car_run}")
    
    """
    chooses an action based on the current state of the enivorment its in. 
    """
    def action_sel(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            state_tensor /= 255.0

            logits, value = self.network(state_tensor)
            distribution = Categorical(logits=logits)

            action = distribution.sample()
            log_probability  = distribution.log_prob(action)

        return int(action.item()), float(log_probability.item()), float(value.item())
    
    """
    Note to self: It checks how likely the selected action is actually according to the current
    policy and estimates the value of each state.
    """
    def evaluate(self, state, action):
        logits, val = self.network(state)
        distribution = Categorical(logits=logits)
        
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy().mean()

        return val, log_probability, entropy
    
    """
    Records information when the epsiode has ended. 
    """
    def finish_ep(self):
        self.ep_number += 1

        print(f"Episode {self.ep_number} | "
              f"Step {self.step_count} | "
              f"Return {self.ep_return}")
        
        self.writer.add_scalar("rollout/episode_return", self.ep_return, self.step_count)
        self.writer.add_scalar("rollout/episode_length", self.ep_length, self.step_count)

        self.recent_returns.append(self.ep_return)
        
        new_obs, info = self.env.reset()
        self.current_state = reset_stack(new_obs)

        self.ep_return = 0.0
        self.ep_length = 0
        

    """
    collects training data by letting the agent to interact with its enviroment 
    """
    def rollout(self):
        state = np.zeros((ROLLOUT_LENGTH, 4, 84, 84), dtype=np.uint8)
        action = np.zeros((ROLLOUT_LENGTH,), dtype=np.int64)
        log_prob = np.zeros((ROLLOUT_LENGTH,), dtype=np.float32)
        rewards = np.zeros((ROLLOUT_LENGTH,), dtype=np.float32)
        flag = np.zeros((ROLLOUT_LENGTH,), dtype=np.float32)
        val = np.zeros((ROLLOUT_LENGTH,), dtype=np.float32)

        for step in range(ROLLOUT_LENGTH):

            self.step_count += 1

            state[step] = self.current_state

            sel_action, log_probability, value = self.action_sel(
                self.current_state
            )

            action[step] = sel_action
            log_prob[step] = log_probability
            val[step] = value

            next_obs, reward, terminated, truncated, info = self.env.step(
                sel_action
            )

            done = terminated or truncated

            rewards[step] = float(reward)
            flag[step] = float(done)

            self.ep_return += reward
            self.ep_length += 1

            self.current_state = update_stack(next_obs)

            if done:
                self.finish_ep()

            if self.step_count >= TOTAL_STEPS:
                break

        return (
            state[:step + 1],
            action[:step + 1],
            log_prob[:step + 1],
            rewards[:step + 1],
            flag[:step + 1],
            val[:step + 1])

    """
    The actually training process of the entire algorithm. It repeatedly collects 
    rollout data and calculates the advantages and makes sure that the network is 
    always updated. 
    """
    def train(self):
        while self.step_count < TOTAL_STEPS:
            states, actions, old_log_probs, rewards, done_flags, values = self.rollout()

            with torch.no_grad():
                current_state_tensor = torch.from_numpy(self.current_state).float().unsqueeze(0).to(self.device)              
                current_state_tensor /= 255.0
                _, final_val = self.network(current_state_tensor)

            advantages, returns = gae_compute(
            list(rewards),
            list(values),
            list(done_flags),
            float(final_val.item()))

            advantages = np.array(advantages, dtype=np.float32)
            returns = np.array(returns, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states_tensor = torch.from_numpy(states).float().to(self.device) / 255.0
            actions_tensor = torch.from_numpy(actions).long().to(self.device)
            old_log_probs_tensor = torch.from_numpy(old_log_probs).float().to(self.device)
            advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
            returns_tensor = torch.from_numpy(returns).float().to(self.device)

            num_samples = len(states_tensor)
            indices = np.arange(num_samples)

            for epoch in range(EPOCHS):
                np.random.shuffle(indices)

                for start in range(0, num_samples, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    batch_indices = indices[start:end]

                    batch_states = states_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_old_log_probs = old_log_probs_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]

                    values_pred, new_log_probs, entropy = self.evaluate(
                        batch_states,
                        batch_actions)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()

                    clip_fraction = ((ratio - 1.0).abs() > CLIP).float().mean()

                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * batch_advantages

                    actor_loss = -torch.min(unclipped, clipped).mean()
                    critic_loss = ((batch_returns - values_pred) ** 2).mean()

                    total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                    self.optimize.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD)
                    self.optimize.step()

            self.writer.add_scalar("train/actor_loss", actor_loss.item(), self.step_count)
            self.writer.add_scalar("train/critic_loss", critic_loss.item(), self.step_count)
            self.writer.add_scalar("train/entropy", entropy.item(), self.step_count)

        self.env.close()
        self.writer.close()
        torch.save(self.network.state_dict(), "ppo_carracing_model.pt")
        print("Training Completed")
        print("model saved as ppo_carraching_model.pt")


def main():
    agent = PPO()
    agent.train()

if __name__ == "__main__":
    main()




        





