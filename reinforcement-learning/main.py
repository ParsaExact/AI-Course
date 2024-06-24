from snake import *
from utility import *
from cube import *

import pygame
import numpy as np
from tkinter import messagebox
from snake import Snake

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT, ))

    snake_1 = Snake((255, 0, 0), (15, 15), SNAKE_1_Q_TABLE)
    snake_2 = Snake((255, 255, 0), (5, 5), SNAKE_2_Q_TABLE)
    rewards_1 = []
    rewards_2 = []
    break_outer = False

    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()

    while True:
        reward_1 = 0
        reward_2 = 0
        # pygame.time.delay(10)
        # clock.tick(1000000) #for training
        clock.tick(15) #for playing
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                if messagebox.askokcancel("Quit", "Do you want to save the Q-tables?"):
                    snake_1.save_q_tabel(SNAKE_1_Q_TABLE)
                    snake_2.save_q_tabel(SNAKE_2_Q_TABLE)
                pygame.quit()
                break_outer = True

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                snake_1.save_q_tabel(SNAKE_1_Q_TABLE)
                snake_2.save_q_tabel(SNAKE_2_Q_TABLE)
                pygame.time.delay(1000)
        if break_outer:
            break
        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
        state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

        snack, reward_1, win_1, win_2 = snake_1.calc_reward(snack, snake_2)
        snack, reward_2, win_2, win_1 = snake_2.calc_reward(snack, snake_1)
        snake_1.update_q_table(state_1, action_1, new_state_1, reward_1)
        snake_2.update_q_table(state_2, action_2, new_state_2, reward_2)

        rewards_1.append(reward_1)
        rewards_2.append(reward_2)
        snake_1.decay_epsilon()
        snake_2.decay_epsilon()
        redrawWindow(snake_1, snake_2, snack, win)

    save_rewards(rewards_1, rewards_2)

if __name__ == "__main__":
    main()
