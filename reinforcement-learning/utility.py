from constants import *
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def redrawWindow(snake_1, snake_2, snack, win):
    win.fill((0, 0, 0))
    drawGrid(WIDTH, ROWS, win)
    snake_1.draw(win)
    snake_2.draw(win)
    snack.draw(win)
    pygame.display.update()
    pass


def drawGrid(w, rows, surface):
    sizeBtwn = w // rows

    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255, 255, 255), (x, 0), (x, w))
        pygame.draw.line(surface, (255, 255, 255), (0, y), (w, y))

    wall_color = (139, 69, 19)  # Brown color
    for i in range(rows):
        pygame.draw.rect(surface, wall_color, (0, i * sizeBtwn, sizeBtwn, sizeBtwn))
        pygame.draw.rect(
            surface,
            wall_color,
            ((rows - 1) * sizeBtwn, i * sizeBtwn, sizeBtwn, sizeBtwn),
        )
        pygame.draw.rect(surface, wall_color, (i * sizeBtwn, 0, sizeBtwn, sizeBtwn))
        pygame.draw.rect(
            surface,
            wall_color,
            (i * sizeBtwn, (rows - 1) * sizeBtwn, sizeBtwn, sizeBtwn),
        )


def randomSnack(rows, item):
    positions = item.body

    while True:
        x = random.randrange(1, rows - 1)
        y = random.randrange(1, rows - 1)
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            continue
        else:
            break

    return (x, y)

def reset(snake_1, snake_2):
    snake_1.reset((5, 5))
    snake_2.reset((15, 15))

def save_rewards(rewards1, rewards2):
    my_data = {"snake_1": rewards1, "snake_2": rewards2}
    df = pd.DataFrame(my_data)
    df.to_csv('my_data.csv', index=False)

def calc_manhattan_distance(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])