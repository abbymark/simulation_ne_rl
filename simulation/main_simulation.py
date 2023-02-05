import pygame
import sys
import random

from units import Unit
from foods import Food

# constants
BACKGROUND_COLOR = (0, 0, 0)
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
FRAME_RATE = 30

FOOD_COUNT = 30

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# create a unit
unit = Unit(100, 100, 50, 50, (255, 0, 0), 5, 100, window)

# create a food at random location
foods = []
for i in range(FOOD_COUNT):
    foods.append(Food(random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT), 10, (0, 255, 0), window))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # draw background
    window.fill(BACKGROUND_COLOR)

    # check if unit is colliding with food
    for food in foods:
        if unit.body.colliderect(food.center[0], food.center[1], food.radius, food.radius):
            unit.health += 10
            food.center = (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT))

    # update
    unit.update(foods)

    # draw
    unit.draw()

    for food in foods:
        food.draw()

    pygame.display.update()

    pygame.time.Clock().tick(FRAME_RATE)