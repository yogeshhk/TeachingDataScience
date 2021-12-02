from typing import List

from decorators import empty

from .library import BinaryTree, Queue


@empty
# Performs a spiral order traversal on the binary tree
def traversal(tree: BinaryTree) -> List[int]:
    # Use a queue to store all nodes that still need to be considered
    queue = Queue()
    # Use a list to store intermediate result of a level
    temp_list = []

    # Flag to check whether we are on odd or even levels
    odd_level = True
    queue.enqueue(tree)

    # List to store final result
    res = []

    while queue:
        # Get next node in the queue
        cur = queue.dequeue()
        # Add the val of the current node to the result
        res.append(cur.val)

        # If we are on an odd level we add left to right,
        # On even levels we add right to left
        # This is the reversed order in which we want the nodes in the result
        if odd_level:
            if cur.has_left():
                temp_list.append(cur.left)
            if cur.has_right():
                temp_list.append(cur.right)
        else:
            if cur.has_right():
                temp_list.append(cur.right)
            if cur.has_left():
                temp_list.append(cur.left)
        # If queue is empty, it means that we are done with the current level
        if not queue:
            # Reverse the list, as we added the nodes in reverse order
            temp_list.reverse()
            # Move all intermediate results to the queue
            for x in temp_list:
                queue.enqueue(x)
            # Reset the temporary list
            temp_list = []
            # Switch to next level
            odd_level = not odd_level
    return res
