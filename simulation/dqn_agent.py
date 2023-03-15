import numpy as np
import torch
import torch.optim as optim

from model import Model
from replay_buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQNAgent:
    def __init__(self, 
                state_size: int,
                action_size: int,
                memory_size: int = 100_000,
                batch_size: int = 64, 
                gamma = 0.9, 
                epsilon_min = 0.1, 
                epsilon_decay = 0.99, 
                learning_rate = 5e-4):

        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.epsilon = 1.0
        self.memory = ReplayBuffer(memory_size, batch_size)

        self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def build_model(self):
        self.model = Model(self.state_size, 24, self.action_size).to(device)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            print("learning", len(self.memory))
            self.learn()
            self.before_episode()
    
    def act(self, state, mode='train'):
        if np.random.rand() <= self.epsilon and mode == 'train':
            return np.random.randint(self.action_size)
        return np.argmax(self.model(state).detach().cpu().numpy())
        
    def before_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        v_s_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)

        q_sa_pure = self.model(states)
        q_sa = q_sa_pure.gather(dim = 1, index = actions)

        # TD = r + gamma * V(s') - Q(s, a)
        td = rewards + self.gamma * v_s_next * (1 - dones) - q_sa

        error = self.loss(td, torch.zeros_like(td))

        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

    
        
