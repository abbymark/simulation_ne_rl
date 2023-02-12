import numpy as np
import torch
import torch.optim as optim

from model import Model

class DQNAgent:
    def __init__(self, 
                state_size: int,
                action_size: int,
                memory_size: int = 100_000,
                batch_size: int = 64, 
                gamma = 0.9, 
                epsilon_min = 0.1, 
                epsilon_decay = 0.9, 
                learning_rate = 5e-4):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate


    def build_model(self):
        self.model = Model(self.state_size, 24, self.action_size)

    def learn(self):
        

