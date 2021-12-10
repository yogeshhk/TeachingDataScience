def num_islands(grid):
    if grid == []:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                count += 1
                DFS(i, j, grid)
    return count
    
def DFS(x, y, grid):
    if (x < 0 or y < 0 or
        x >= len(grid) or y >= len(grid[0]) or
        grid[x][y] != "1"):
        return
    grid[x][y] = "0"
    DFS(x + 1, y, grid)
    DFS(x - 1, y, grid)
    DFS(x, y + 1, grid)
    DFS(x, y - 1, grid)

print(num_islands([
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]))