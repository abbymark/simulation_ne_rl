import pygame

class Food:
    def __init__(self, x, y, radius, color, window):
        # body should be circle
        self.center = (x, y)
        self.radius = radius
        self.color = color
        self.window = window

    def draw(self):
        pygame.draw.circle(self.window, self.color, self.center, self.radius)