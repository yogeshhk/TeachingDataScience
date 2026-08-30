# Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[
# i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day
# for which this is possible, keep answer[i] == 0 instead.
#
# Example 1:
#
# Input: temperatures = [73,74,75,71,69,72,76,73]
# Output: [1,1,4,2,1,1,0,0]
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = list() # of (temp, index)

    for i, t in enumerate(temperatures):
        while stack and t > stack[-1][0]: # -1 is for top of the stack, and 0 for first of the tuple
            stack_temp, stack_indx = stack.pop()
            result[stack_indx] = (i - stack_indx)
        stack.append([t,i])
    return result

temps = [73,74,75,71,69,72,76,73]
result = daily_temperatures(temps)
print(result)
