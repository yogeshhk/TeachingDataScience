class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def level_avg(root):
    averages = []
    queue = []
    queue.append(root)
        
    while (queue):
        count = len(queue)
        level_sum = 0
        for _ in range(count):
            node = queue.pop(0)
            level_sum += node.val
            if (node.left):
                queue.append(node.left)
            if (node.right):
                queue.append(node.right)
        averages.append(level_sum / float(count))
    return averages

'''
      3
  9      20
       15   7
'''

n1 = Node(3)
n2 = Node(9)
n3 = Node(20)
n4 = Node(15)
n5 = Node(7)

n1.left = n2
n1.right = n3
n3.left = n4
n3.right = n5

print(level_avg(n1)) # [3.0, 14.5, 11.0]