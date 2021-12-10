
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dict = {}
        # Init the dummy nodes
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key not in self.dict:
            return -1
        # Re-add the node
        node = self.dict[key]
        self.remove(node)
        self.add(node)
        return node.val

    def put(self, key, value):
        if key in self.dict:
            self.remove(self.dict[key])
        node = Node(key, value)
        self.add(node)
        self.dict[key] = node
        if len(self.dict) > self.capacity:
            node = self.head.next
            self.remove(node)
            del self.dict[node.key]

    # Remove the node
    def remove(self, node):
        prevNode = node.prev
        nextNode = node.next
        prevNode.next = nextNode
        nextNode.prev = prevNode

    # Add the node to the end
    def add(self, node):
        prevNode = self.tail.prev
        prevNode.next = node
        self.tail.prev = node
        node.prev = prevNode
        node.next = self.tail

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))       # returns 1
cache.put(3, 3)           # evicts key 2
print(cache.get(2))       # returns -1 (not found)
cache.put(4, 4)           # evicts key 1
print(cache.get(1))       # returns -1 (not found)
print(cache.get(3))       # returns 3
print(cache.get(4))       # returns 4