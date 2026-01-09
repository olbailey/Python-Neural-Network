import pygame
import pandas as pd
import numpy as np

def draw_image(data):
    pygame.init()
    screen = pygame.display.set_mode((448, 448)) #(448, 448) = 16x16 pixel ratio
    pygame.display.set_caption("Number Drawer")
    
    SIZE = 28
    RATIO = 16
    x, y = 0, 0
    for value in data:
        colour = [value for _ in range(3)]

        pygame.draw.rect(screen, colour, (x*RATIO, y*RATIO, RATIO, RATIO))
        if x + 1 < SIZE:
            x += 1
        else:
            x = 0
            if y < SIZE:
                y += 1
            else:
                raise Exception('shit got real')
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

df = pd.read_csv('mnist_train.csv')
data = np.array(df.iloc[0])[1:]

draw_image(data)