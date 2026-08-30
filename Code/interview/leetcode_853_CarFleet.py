# There are n cars going to the same destination along a one-lane road. The destination is target miles away.
#
# You are given two integer array position and speed, both of length n, where position[i] is the position of the ith
# car and speed[i] is the speed of the ith car (in miles per hour).
#
# A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same
# speed. The faster car will slow down to match the slower car's speed. The distance between these two cars is
# ignored (i.e., they are assumed to have the same position).
#
# A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is
# also a car fleet.
#
# If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.
#
# Return the number of car fleets that will arrive at the destination.
#
#
#
# Example 1:
#
# Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3] Output: 3 Explanation: The cars starting at 10 (
# speed 2) and 8 (speed 4) become a fleet, meeting each other at 12. The car starting at 0 does not catch up to any
# other car, so it is a fleet by itself. The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet,
# meeting each other at 6. The fleet moves at speed 1 until it reaches target. Note that no other cars meet these
# fleets before the destination, so the answer is 3.

# Hint, position-speed pair can be plotted. Intersections make them car fleet, merging with slower one.
# Find time to destination for each. So one which is taking lesser, they will merge. Go from right to left, after
# sorting by position

def car_fleet(target, position,speed):
    pair = [[p,s] for p,s in zip(position,speed)]
    stack = []
    for p,s in sorted(pair)[::-1]: # Reverse sorted order
        stack.append((target-p) /s) # time-to-destination
        # Does it overlap?? less time to reach
        if len(stack) >= 2 and stack[-1] <= stack[-2]: # why not while loop? as we are traversing from right, not needed
            stack.pop()
    return len(stack)

