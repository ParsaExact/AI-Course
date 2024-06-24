import random
import numpy as np
import copy
import pickle
from collections import defaultdict

from cube import Cube
from constants import *
from utility import *


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            with open("Qtabel"+file_name+".pickle", "rb") as file:
                self.q_table = defaultdict(lambda: [0,0,0,0], pickle.load(file))
        except:
            self.q_table = defaultdict(lambda: [0,0,0,0])

        self.lr = LEARNING_RATE #  Learning rate
        self.discount_factor = DISCOUNT_FACTOR #  Discount factor
        self.epsilon = INITIAL_EPSILON #  Epsilon
        self.epsilon_decay = EPSILON_DECAY

    def decay_epsilon(self):
        if self.epsilon <= EPSILON_MIN:
            self.epsilon = EPSILON_MIN
        self.epsilon *= self.epsilon_decay

    def get_optimal_policy(self, state):
        """Get optimal policy."""
        sorted_indices = np.argsort(self.q_table[state])
        sorted_indices = sorted_indices[::-1]
        return sorted_indices[0], sorted_indices[1]

    def find_direction(self, dx, dy):
        """This function determines direction of the move."""
        if dx == 1 and dy == 0:
            return RIGHT
        if dx == -1 and dy == 0:
            return LEFT
        if dx == 0 and dy == -1:
            return UP
        if dx == 0 and dy == 1:
            return DOWN

    def is_opposite_direction(self, next_dir):
        """This function returns True if the given direction is exactly opposite of the Snake's direction."""
        direction = self.find_direction(self.dirnx, self.dirny)
        return (direction == LEFT and next_dir == RIGHT) | \
               (direction == RIGHT and next_dir == LEFT) | \
               (direction == UP and next_dir == DOWN) | \
               (direction == DOWN and next_dir == UP)

    def make_action(self, state):
        """Choose the action with highest sample reward and the one that is not backwards."""
        chance = random.random()
        if chance < self.epsilon :
            action = random.randint(0, 3)
        else:
            action1 , action2 = self.get_optimal_policy(state)
            action = action1 if self.is_opposite_direction(next_dir=action1) == False else action2

        return action

    def update_q_table(self, state, action, next_state, reward):
        # Update Q-table
        if state not in self.q_table:
            my_state = sum(i*int(state[i]) for i in range(4))
            self.q_table[state][my_state] = 1
        if next_state not in self.q_table:
            my_next_state = sum(i*int(next_state[i]) for i in range(4))
            self.q_table[next_state][my_next_state] = 1
        self.q_table[state][action] = self.q_table[state][action] + \
        self.lr * (reward  + \
        self.discount_factor * np.max(self.q_table[next_state]) -\
        self.q_table[state][action])

    def snack_state(self, snack):
        relative_food_position = [0,0,0,0,0,0]
        if (snack.pos[0] - self.head.pos[0]) > 0:        #foodRight
            relative_food_position[0] = 1
        if (snack.pos[0] - self.head.pos[0]) < 0 :       #foodLeft
            relative_food_position[1] = 1
        if snack.pos[0] - self.head.pos[0] == 0:     #foodXMiddle
            relative_food_position[2] = 1

        return ''.join(map(str, relative_food_position))

    def edge_state(self,distnace):
        screen_danger = [0,0,0,0]
        if(self.head.pos[0] <= 1 + distnace): #dangerRight
            screen_danger[0] = 1
        if(self.head.pos[0] >= 18 - distnace): #dangerLeft
            screen_danger[1] = 1
        if(self.head.pos[1] <= 1 + distnace): #dangerBottom
            screen_danger[2] = 1
        if(self.head.pos[1] >= 18 - distnace): #dangerTop
            screen_danger[3] = 1

        return ''.join(map(str, screen_danger))

    def enemy_state(self, size, other_snake):
        """Gives the states of a 3*3 square over the snake head."""
        distance = (size - 1) // 2
        offsets = np.union1d(np.arange(-distance, distance + 1), np.arange(distance + 1))
        states_of_neighbors = []

        for dx in offsets:
            for dy in offsets:
                neighbor_pos = (self.head.pos[0] + dx, self.head.pos[1] + dy)
                if neighbor_pos == other_snake.head.pos or \
                   neighbor_pos in [z.pos for z in other_snake.body if z.pos != other_snake.head.pos] or \
                   neighbor_pos in [z.pos for z in self.body[1:]] or \
                   not (1 <= neighbor_pos[0] < ROWS - 1 and 1 <= neighbor_pos[1] < ROWS - 1):
                    states_of_neighbors.append(0)
                else:
                    states_of_neighbors.append(1)

        return ''.join(map(str, states_of_neighbors))


    def my_snake_state(self, radius, obstacles):
        offsets = [(dx, dy) for dx in range(-radius, radius+1) for dy in range(-radius, radius+1) if (dx or dy) and abs(dx) + abs(dy) <= radius]
        density_count = {i: 0 for i in range(1, 5)}

        for offset_x, offset_y in offsets:
            new_x, new_y = (self.head.pos[0] + offset_x, self.head.pos[1] + offset_y)
            if (new_x, new_y) in obstacles:
                quadrant = 1 if offset_x >= 0 and abs(offset_x) >= abs(offset_y) else \
                           3 if offset_x <= 0 and abs(offset_x) >= abs(offset_y) else \
                           2 if offset_y >= 0 and abs(offset_y) >= abs(offset_x) else \
                           4 if offset_y <= 0 and abs(offset_y) >= abs(offset_x) else 0
                if quadrant:
                    density_count[quadrant] += 1
        return max(density_count, key=density_count.get) if any(density_count.values()) else 0

    def create_state(self, snack, other_snake):
        states_snack = self.snack_state(snack)
        state_neighbor = self.enemy_state(NEIGHBOR_RADIUS, other_snake=other_snake)
        states_edge = self.edge_state(distnace=2)
        state_enemy = self.my_snake_state(radius=3, obstacles=list(map(lambda z: z.pos, self.body[1:])))
        states = states_snack +  "_" + str(state_enemy) + "_" + state_neighbor + "_" + states_edge
        return states

    def move(self, snack, other_snake):
        self.pre_head = copy.deepcopy(self.head)
        state = self.create_state(snack,other_snake)# Create state
        action = self.make_action(state)
        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
        next_state = self.create_state(snack,other_snake)

        return state, next_state, action

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            # Punish the snake for getting out of the board
            reward += KILL
            win_other = True
            reset(self, other_snake)

        if self.head.pos == snack.pos:
            # Reward the snake for eating
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += SNACK_REWARD

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # Punish the snake for hitting itself
            reward += KILL
            win_other = True
            reset(self, other_snake)

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                # Punish the snake for hitting the other snake
                reward += KILL
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    # Reward the snake for hitting the head of the other snake and being longer
                    reward += WIN
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    # No winner
                    reward += TIE
                else:
                    # Punish the snake for hitting the head of the other snake and being shorter
                    reward += KILL
                    win_other = True

            reset(self, other_snake)
        if calc_manhattan_distance(self.pre_head.pos,snack.pos) > \
            calc_manhattan_distance(self.head.pos,snack.pos):
            reward += CLOSER_TO_SNACK
        else:
            reward += FARTHER_TO_SNACK
        # Reward the snake if it is getting closer to snack

        reward += 1 / (calc_manhattan_distance(self.head.pos,(CENTER_X,CENTER_Y)) + \
            (calc_manhattan_distance(self.head.pos,(CENTER_X,CENTER_Y)) == 0))
        # punish the snake based on its closeness to border lines

        return snack, reward, win_self, win_other

    def save_q_tabel(self,num):
        with open("Qtabel"+num+".pickle","wb" ) as file:
                pickle.dump(dict(self.q_table), file)

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)
