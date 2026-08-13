# https://www.geeksforgeeks.org/egg-dropping-puzzle-dp-11/
# Suppose that we wish to know which stories in a 36-story building are safe to drop eggs from, and which will cause
# the eggs to break on landing. We make a few assumptions:
# …..An egg that survives a fall can be used again.
# …..A broken egg must be discarded.
# …..The effect of a fall is the same for all eggs.
# …..If an egg breaks when dropped, then it would break if dropped from a higher floor.
# …..If an egg survives a fall then it would survive a shorter fall.
# …..It is not ruled out that the first-floor windows break eggs, nor is it ruled out that
# the 36th-floor do not cause an egg to break.
# If only one egg is available and we wish to be sure of obtaining the right result, the experiment can be carried
# out in only one way. Drop the egg from the first-floor window; if it survives, drop it from the second-floor window.
# Continue upward until it breaks. In the worst case, this method may require 36 droppings. Suppose 2 eggs are available.
# What is the least number of egg-droppings that is guaranteed to work in all cases?
# The problem is not actually to find the critical floor, but merely to decide floors from which eggs should be dropped
# so that the total number of trials are minimized.

#  The solution is to try dropping an egg from every floor(from 1 to k) and recursively calculate the minimum number
#  of droppings needed in the worst case. The floor which gives the minimum value in the worst case is going to be part
#  of the solution.
# In the following solutions, we return the minimum number of trials in the worst case; these solutions can be easily
# modified to print floor numbers of every trial also.
# Meaning of a worst-case scenario: Worst case scenario gives the user the surety of the threshold floor. For example-
# If we have ‘1’ egg and ‘k’ floors, we will start dropping the egg from the first floor till the egg breaks suppose
# on the ‘kth’ floor so the number of tries to give us surety is ‘k’.

def egg_drop(n,k):
    # If there are no floors, then no trials
    # needed. OR if there is one floor, one
    # trial needed.
    if (k == 1 or k == 0):
        return k

    # We need k trials for one egg
    # and k floors
    if (n == 1):
        return k

    global_min = 100000

    for x in range(1,k+1):
        result = max(egg_drop(n-1,x-1), egg_drop(n, k -x))
        if result < global_min:
            global_min = result
    return global_min + 1

n = 2
k = 10
print("Minimum number of trials in worst case with", n, "eggs and", k, "floors is", egg_drop(n, k))