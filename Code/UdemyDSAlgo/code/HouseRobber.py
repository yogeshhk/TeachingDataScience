#   Created by Elshad Karimov 
#   Copyright © AppMillers. All rights reserved.

# House Robber Problem  in Python

def houseRobber(houses, currentIndex):
    if currentIndex >= len(houses):
        return 0
    else:
        stealFirstHouse = houses[currentIndex] + houseRobber(houses, currentIndex + 2)
        skipFirstHouse = houseRobber(houses, currentIndex+1)
        return max(stealFirstHouse, skipFirstHouse)

houses = [6,7,1,30,8,2,4]
print(houseRobber(houses, 0))