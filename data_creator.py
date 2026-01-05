import pygame
import numpy as np
import pandas as pd
import csv
import random

def run():
    def draw_display():
        screen.fill(BLACK)
        pygame.draw.rect(screen, GREY, (0, 0, D_WIDTH, D_HEIGHT), MARGIN)
        pygame.draw.rect(screen, WHITE, (MARGIN, MARGIN, AXIS_WIDTH, D_HEIGHT - MARGIN*2))
        pygame.draw.rect(screen, WHITE, (MARGIN, D_HEIGHT-MARGIN, D_WIDTH - MARGIN*2, AXIS_WIDTH))

    def draw_points(array):
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                value = array[y][x]
                if value > 0:
                    if value == 1:
                        colour = BLUE
                    else:
                        colour = RED
                    # pygame.draw.rect(screen, (RED if colour == BLUE else BLUE), (MARGIN+AXIS_WIDTH+POINT_SIZE*x, D_HEIGHT-MARGIN-AXIS_WIDTH-POINT_SIZE*y-POINT_SIZE/2, POINT_SIZE, POINT_SIZE))
                    pygame.draw.circle(screen, (RED if colour == BLUE else BLUE), (MARGIN+AXIS_WIDTH+POINT_SIZE*x+POINT_SIZE/2, D_HEIGHT-MARGIN-AXIS_WIDTH-POINT_SIZE*y+POINT_SIZE/2), 1)

    def convert_mpos(pos):
        new_pos = (pos[0]-MARGIN-AXIS_WIDTH, D_HEIGHT-MARGIN - AXIS_WIDTH - pos[1])
        new_pos = (new_pos[0]//POINT_SIZE, new_pos[1]//POINT_SIZE)
        return new_pos

    def assign_datapoint(pos, colour_id):
        if not(MARGIN + AXIS_WIDTH <= pos[0] < D_WIDTH - MARGIN and MARGIN < pos[1] < D_HEIGHT - MARGIN - AXIS_WIDTH):
            return

        array_pos = convert_mpos(pos)
        grid[array_pos[1]][array_pos[0]] = colour_id

    def populate_datapoints(array, multiplier, spread) -> np.ndarray:
        new_array = array.copy()
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                value = array[y][x]
                if value > 0:
                    new_array[y][x] = value
                    for _ in range(multiplier):
                        dx = random.randint(-spread, spread)
                        dy = random.randint(-spread, spread)

                        if 0 <= y + dy < array.shape[0] and 0 <= x + dx < array.shape[1]:
                            new_array[y+dy][x+dx] = value
        return new_array

    def convert_data(array):
        data = [['label', 'x', 'y']]
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                value = array[y][x]
                if value > 0:
                    data.append([value, x, y])

        return data

    def create_csv(array):
        name = input('Enter name of file: ')
        with open(name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(convert_data(array))

    D_WIDTH = 800
    D_HEIGHT = 800
    MARGIN = 25
    AXIS_WIDTH = 2
    POINT_SIZE = 4

    # Colours
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREY = (25, 25, 25)

    # multiplier vars
    SPREAD = 10
    MULTIPLIER = 5

    pygame.init()
    screen = pygame.display.set_mode((D_WIDTH, D_HEIGHT)) #(448, 448) = 16x16 pixel ratio
    pygame.display.set_caption("Data Creator")

    draw_display()
    grid = np.zeros(((D_HEIGHT-MARGIN*2)//POINT_SIZE, (D_WIDTH-MARGIN*2)//POINT_SIZE), np.int8)
    print(grid.shape)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            button_type = pygame.mouse.get_pressed()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                print('here')
                if button_type[0]:
                    print('1')
                    assign_datapoint(mouse_pos, 1)
                elif button_type[2]:
                    print('2')
                    assign_datapoint(mouse_pos, 2)
                    
                draw_points(grid)

            if button_type[1]:
                mouse_pos = pygame.mouse.get_pos()
                draw_display()
                assign_datapoint(mouse_pos, 0)

                draw_points(grid)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    create_csv(grid)
                elif event.key == pygame.K_m:
                    grid = populate_datapoints(grid, MULTIPLIER, SPREAD)
                    print('multiplied')
                    draw_points(grid)

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run()