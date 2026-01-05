import numpy as np
import pygame

def DrawbrushPixels():
    for y in range(0, 441, 8):
        for x in range(0, 441, 8):
            gridColourStrength = gridMatrix[y//8][x//8]
            if gridMatrix[y//8][x//8] > 0:
                pygame.draw.rect(screen, (gridColourStrength, gridColourStrength, gridColourStrength), pygame.Rect(x, y, pixelSize, pixelSize))
def DrawMnistPixels():
    for y in range(0, 433, 16):
        for x in range(0, 433, 16):
            gridColourStrength = MnistPictureGrid[y//16][x//16]
            if MnistPictureGrid[y//16][x//16] > 0:
                pygame.draw.rect(screen, (gridColourStrength, gridColourStrength, gridColourStrength), pygame.Rect(x, y, 16, 16))

def FindAverageColourOfPixel():
    for y in range(0, 28):
        for x in range(0, 28):
            sumColour = 0
            for y2 in range(0, 2):
                for x2 in range(0, 2):
                    sumColour += gridMatrix[2*y + y2][2*x + x2]
            MnistPictureGrid[y][x] = int(sumColour // 4)



pygame.init()
screen = pygame.display.set_mode((448, 448)) #(448, 448) = 16x16 pixel ratio
pygame.display.set_caption("Number Drawer")

pictureStorage = []
enteredNumbers = []
gridMatrix = np.zeros((56, 56))
MnistPictureGrid = np.zeros((28, 28))
pixelSize = 8
displayingMnist = False

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if not displayingMnist:
            if pygame.mouse.get_pressed()[0]:
                mousePos = pygame.mouse.get_pos()
                if 440 > mousePos[0] > 8 and 440 > mousePos[1] > 8:
                    if gridMatrix[mousePos[1]//8][mousePos[0]//8] < 255:
                        gridMatrix[mousePos[1]//8][mousePos[0]//8] = 255
                    # one up
                    if gridMatrix[(mousePos[1]//8) - 1][(mousePos[0]//8)] + 127.5 > 255:
                        gridMatrix[(mousePos[1]//8) - 1][(mousePos[0]//8)] == 255
                    else:
                        gridMatrix[(mousePos[1]//8) - 1][(mousePos[0]//8)] += 127.5
                    # one down
                    if gridMatrix[(mousePos[1]//8) + 1][(mousePos[0]//8)] + 127.5 > 255:
                        gridMatrix[(mousePos[1]//8) + 1][(mousePos[0]//8)] == 255
                    else:
                        gridMatrix[(mousePos[1]//8) + 1][(mousePos[0]//8)] += 127.5
                    # one left
                    if gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) - 1] + 127.5 > 255:
                        gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) - 1] == 255
                    else:
                        gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) - 1] += 127.5
                    # one right
                    if gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) + 1] + 127.5 > 255:
                        gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) + 1] == 255
                    else:
                        gridMatrix[(mousePos[1]//8)][(mousePos[0]//8) + 1] += 127.5
            elif event.type == pygame.KEYDOWN:
                displayingMnist = True
                FindAverageColourOfPixel()
                pictureStorage.append(MnistPictureGrid)
                #enteredNumber = int(input('What number did you draw: '))
                #enteredNumbers.append(enteredNumber)
        else:
            if event.type == pygame.KEYDOWN:
                displayingMnist = False
                gridMatrix = np.zeros((56, 56))

            

    screen.fill((0, 0, 0))
    if not displayingMnist:
        DrawbrushPixels()  
    else:
        DrawMnistPixels() 

    pygame.display.flip()

pygame.quit()

exposedPictureStorage  = []

for row in pictureStorage[0]:
    for pixel in row:
        exposedPictureStorage.append(pixel)

# print(exposedPictureStorage)