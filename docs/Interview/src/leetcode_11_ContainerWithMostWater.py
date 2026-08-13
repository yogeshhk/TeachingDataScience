# Build histogram of numbers. With two bars as sides, find which container has most water
# Example [1,8,6,2,5,4,8,3,7] = 49, between first and last

def maxArea(height):
    start_index = 0
    end_index = len(height) - 1
    largest = 0
    while start_index != end_index:
        next_area = min(height[start_index],height[end_index]) * (end_index - start_index)
        if next_area > largest:
            largest = next_area
        if height[start_index] < height[end_index]:
            start_index += 1
        else:
            end_index -= 1
    return largest
