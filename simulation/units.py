import pygame
import torch
from scipy.spatial import cKDTree
from enum import Enum

from model import Model


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
        self.model = Model(len(InputTypes), 10, len(OutputTypes))

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
        elif self.body.x > self.window.get_width() - self.body.width:
            self.body.x = self.window.get_width() - self.body.width
        if self.body.y < 0:
            self.body.y = 0
        elif self.body.y > self.window.get_height() - self.body.height:
            self.body.y = self.window.get_height() - self.body.height

    def update(self, foods):
        # get nearest food
        nearest_food = self.get_nearest_food(foods)
        with torch.no_grad():
            self.model.eval()
            input_list = [
                float(self.body.x),
                float(self.body.y),
                float(self.health),
                float(nearest_food.center[0]),
                float(nearest_food.center[1])
            ]
            output = self.model(torch.tensor(input_list))
            print(output)

        # move unit based on output argmax
        self.move(output.argmax())

        # check if unit is out of bounds
        self.check_bounds()

        # decrease health
        self.health -= 0.1

    def draw(self):
        pygame.draw.rect(self.window, self.color, self.body)
