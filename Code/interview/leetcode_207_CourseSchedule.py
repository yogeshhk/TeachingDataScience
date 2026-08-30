# There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array
# prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take
# course ai.
#
# For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
# Return true if you can finish all courses. Otherwise, return false.
#
#
#
# Example 1:
#
# Input: numCourses = 2, prerequisites = [[1,0]]
# Output: true
# Explanation: There are a total of 2 courses to take.
# To take course 1 you should have finished course 0. So it is possible.
# Prepare Adjacency List. Do DFS on start node and see if you can reach last node. All in the path, can be completed.

# n = 5, so nodes are from 0 to 4
# Prerequisites = [[0,1],[0,2],[1,3],[1,4],[3,4]]

# Adjacency List = { 0: [1,2], 1:[3,4], 2:[], 3:[4], 4:[]} # outgoing edges

def can_finish(numCourses, prerequisites):
    adjacency_map = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        adjacency_map[course].append(prereq)

        visited = set()

        def dfs(crs):
            if crs in visited:
                return False
            if not adjacency_map[crs]:
                return True

            visited.add(crs)
            for pre in adjacency_map[crs]:
                if not dfs(pre): return False
            visited.remove(crs)
            adjacency_map[crs] = []
            return True
    for crs in range(numCourses):
        if not dfs(crs): return False
    return True


