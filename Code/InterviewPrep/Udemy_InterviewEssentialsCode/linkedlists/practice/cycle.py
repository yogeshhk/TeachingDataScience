class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def has_cycle(head):
    if head == None:
        return False
    slow = head
    fast = head.next

    while (fast != None and fast.next != None and slow != None):
        if fast == slow:
            return True
        fast = fast.next.next
        slow = slow.next
    return False

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)
n6 = Node(6)
n1.next = n2
n2.next = n3
n3.next = n4
n4.next = n5
n5.next = n6

print(has_cycle(n1))
n4.next = n2
print(has_cycle(n1))
