# https://www.geeksforgeeks.org/flood-fill-algorithm-implement-fill-paint/
# In MS-Paint, when we take the brush to a pixel and click, the color of the region of that pixel is replaced
# with a new selected color. Following is the problem statement to do this task.
#
# Given a 2D screen, location of a pixel in the screen and a color, replace color of the given pixel and
# all adjacent same colored pixels with the given color.
#
# Example:
#
# Input:
# screen[M][N] = {{1, 1, 1, 1, 1, 1, 1, 1},
#                {1, 1, 1, 1, 1, 1, 0, 0},
#                {1, 0, 0, 1, 1, 0, 1, 1},
#                {1, 2, 2, 2, 2, 0, 1, 0},
#                {1, 1, 1, 2, 2, 0, 1, 0},
#                {1, 1, 1, 2, 2, 2, 2, 0},
#                {1, 1, 1, 1, 1, 2, 1, 1},
#                {1, 1, 1, 1, 1, 2, 2, 1},
#                };
#     x = 4, y = 4, newColor = 3
# The values in the given 2D screen
#   indicate colors of the pixels.
# x and y are coordinates of the brush,
#    newColor is the color that
# should replace the previous color on
#    screen[x][y] and all surrounding
# pixels with same color.
M = 8
N = 8
def flood_fill(screen, x,y,new_color, previous_color):
    if x < 0 or y < 0 or x >= M or y >= N or screen[x][y] != previous_color or screen[x][y] == new_color:
        return
    screen[x][y] = new_color
    flood_fill(screen, x+1, y, new_color, previous_color)
    flood_fill(screen, x-1, y, new_color, previous_color)
    flood_fill(screen, x, y+1, new_color, previous_color)
    flood_fill(screen, x, y-1, new_color, previous_color)


def driver(screen,x,y,new_color):
    previous_color = screen[x][y]
    if previous_color == new_color:
        return
    flood_fill(screen,x,y,new_color,previous_color)


screen = [[1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 0, 0],
          [1, 0, 0, 1, 1, 0, 1, 1],
          [1, 2, 2, 2, 2, 0, 1, 0],
          [1, 1, 1, 2, 2, 0, 1, 0],
          [1, 1, 1, 2, 2, 2, 2, 0],
          [1, 1, 1, 1, 1, 2, 1, 1],
          [1, 1, 1, 1, 1, 2, 2, 1]]

x = 4
y = 4
newC = 3
driver(screen, x, y, newC)

print("Updated screen after call to floodFill:")
for i in range(M):
    for j in range(N):
        print(screen[i][j], end=' ')
    print()
