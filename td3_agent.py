import numpy as np
import random
import copy
from collections import namedtuple, deque

from td3_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # Update Every
WARMUP = 1000           # Warm up time step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic One Network (w/ Target Network)
        self.critic_local_one = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target_one = Critic(state_size, action_size, random_seed).to(device)
        self.critic_one_optimizer = optim.Adam(self.critic_local_one.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Critic Two Network (w/ Target Network)
        self.critic_local_two = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target_two = Critic(state_size, action_size, random_seed).to(device)
        self.critic_two_optimizer = optim.Adam(self.critic_local_two.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        #
        # # Noise process
        # self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Counter 
        self.t_step = 0
        
        # learn_counter
        self.learn_ctr = 0





    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #
        # if self.t_step == 0:
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += np.random.normal(0, 0.2, size=action.shape)
        return np.clip(action, -1, 1)

#     def act(self, state, add_noise=True):
#         """Returns actions for given state as per current policy."""

#         if self.t_step < WARMUP:
#             action = np.random.normal(scale=0.1, size=(self.action_size))
#         else:
#             state = torch.from_numpy(state).float().to(device)
#             self.actor_local.eval()
#             with torch.no_grad():
#                 action = self.actor_local(state).cpu().data.numpy()
#             self.actor_local.train()
#             if add_noise:
#                 action += np.random.normal(0, 0.1, action.shape)
       
#         #update counter
#         self.t_step += 1

#         return np.clip(action, -1, 1)



    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        noise = torch.randn_like(actions_next).mul(0.2)
        noise = noise.clamp(-0.5, 0.5)
        actions_next = (actions_next + noise).clamp(-1, 1)

        # actions_next = self.actor_target(next_states)
        # actions_next = actions_next + torch.clamp(torch.from_numpy(np.random.normal(loc=0, scale=0.2, size=actions_next.shape)).float().to(device), -0.5, 0.5)
        # actions_next = torch.clamp(actions_next, self.min_size[0], self.max_size[0])

        critic_one_target = self.critic_target_one(next_states, actions_next)
        critic_two_target = self.critic_target_two(next_states, actions_next)

        Q_targets_next = torch.min(critic_one_target, critic_two_target)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets.detach()

        critic_one_expected = self.critic_local_one(states, actions)
        critic_two_expected = self.critic_local_two(states, actions)

        # Compute Q targets for current states (y_i)



        # Compute both critics loss
        # Minimize the loss


        critic_one_loss = F.mse_loss(critic_one_expected, Q_targets)
        critic_two_loss = F.mse_loss(critic_two_expected, Q_targets)
        critic_loss = critic_one_loss + critic_two_loss

        self.critic_one_optimizer.zero_grad()
        self.critic_two_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_one_optimizer.step()
        self.critic_two_optimizer.step()

        self.learn_ctr = (self.learn_ctr + 1) % UPDATE_EVERY

        if self.learn_ctr != 0:
            return

        # ---------------------------- update actor ---------------------------- #
        
        # Compute actor loss
        actor_loss = -self.critic_local_one(states, self.actor_local(states)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local_one, self.critic_target_one, TAU)
        self.soft_update(self.critic_local_two, self.critic_target_two, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)   

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# class OUNoise:
#     """Ornstein-Uhlenbeck process."""
#
#     def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
#         """Initialize parameters and noise process."""
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.seed = random.seed(seed)
#         self.reset()
#
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)
#
#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
#         self.state = x + dx
#         return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)