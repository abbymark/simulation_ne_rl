import pygame
import torch
from scipy.spatial import cKDTree
from enum import Enum
import numpy as np

from model import Model
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputTypes(Enum):
    X = 0
    Y = 1
    HEALTH = 2
    NEAREST_FOOD_X = 3
    NEAREST_FOOD_Y = 4


class OutputTypes(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Unit:
    def __init__(self, x: int, y: int, width, height, color, speed, health, window):
        self.body = pygame.Rect(x, y, width, height)
        self.color = color

        self.speed = speed
        self.health = health
        self.window = window

        # create model
        self.agent = DQNAgent(state_size=5, action_size=4)

        self.is_alive = True

    def get_nearest_food(self, foods):
        # create a kdtree
        kdtree = cKDTree([(food.center[0], food.center[1]) for food in foods])

        # get nearest food
        nearest_food_distance, nearest_food_index = kdtree.query((self.body.x, self.body.y), k=1, p=1)

        # return nearest food
        return foods[nearest_food_index]

    def move(self, direction):
        if direction == 0:
            self.body.x -= self.speed
        elif direction == 1:
            self.body.x += self.speed
        elif direction == 2:
            self.body.y -= self.speed
        elif direction == 3:
            self.body.y += self.speed

    def check_bounds(self):
        # check if unit is out of bounds
        if self.body.x < 0:
            self.body.x = 0
            self.health -= 0.2
        elif self.body.x > self.window.get_width() - self.body.width:
            self.body.x = self.window.get_width() - self.body.width
            self.health -= 0.2
        if self.body.y < 0:
            self.body.y = 0
            self.health -= 0.2
        elif self.body.y > self.window.get_height() - self.body.height:
            self.body.y = self.window.get_height() - self.body.height
            self.health -= 0.2

    def update(self, foods):
        # get nearest food
        nearest_food = self.get_nearest_food(foods)
        
        # get state
        state = np.array([
            self.body.x,
            self.body.y,
            self.health,
            nearest_food.center[0],
            nearest_food.center[1]
        ], dtype=np.float32)

        # get action
        action = self.agent.act(torch.tensor(state, dtype=torch.float32).to(device))

        # move unit based on output argmax
        self.move(action)

        # check if unit is out of bounds
        self.check_bounds()

        # decrease health
        self.health -= 0.01
        print(self.health)

        # check if unit is dead
        if self.health <= 0:
            self.is_alive = False

        # next state
        new_nearest_food = self.get_nearest_food(foods)
        next_state = np.array([
            self.body.x,
            self.body.y,
            self.health,
            new_nearest_food.center[0],
            new_nearest_food.center[1]
        ], dtype=np.float32)

        # step agent
        self.agent.step(state, action, self.health, next_state, not self.is_alive)
 

    def draw(self):
        pygame.draw.rect(self.window, self.color, self.body)
