class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def print_list(l):
    cur = l
    while (cur != None):
        print(str(cur.val) + " -> ", end='')
        cur = cur.next
    print("None")

def merge(l1, l2):
    # Dummy node technique
    dummy = Node(0)
    current = dummy
    while (l1 != None and l2 != None):
        if l1.val < l2.val:
            current.next = Node(l1.val)
            l1 = l1.next
        else:
            current.next = Node(l2.val)
            l2 = l2.next
        current = current.next
        
    # Get any leftover nodes
    while (l1 != None):
        current.next = Node(l1.val)
        l1 = l1.next
        current = current.next
    while (l2 != None):
        current.next = Node(l2.val)
        l2 = l2.next
        current = current.next
            
    return dummy.next


n1 = Node(1)
n2 = Node(3)
n3 = Node(5)
n4 = Node(7)
n1.next = n2
n2.next = n3
n3.next = n4

n5 = Node(2)
n6 = Node(4)
n7 = Node(6)
n8 = Node(8)
n9 = Node(10)
n10 = Node(12)
n5.next = n6
n6.next = n7
n7.next = n8
n8.next = n9
n9.next = n10

print_list(n1)
print_list(n5)
print_list(merge(n1, n5))
