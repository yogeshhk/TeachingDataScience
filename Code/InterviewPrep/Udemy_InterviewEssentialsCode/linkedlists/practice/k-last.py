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

n1 = Node(3)
n2 = Node(1)
n3 = Node(5)
n4 = Node(2)
n1.next = n2
n2.next = n3
n3.next = n4


# Can assume input is valid
def kth_last(head, k):
    count = 0
    first = head
    while (count != k):
        first = first.next
        count += 1

    second = head
    while (first != None):
        first = first.next
        second = second.next
    return second


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

print_list(n1)

for i in range(1, 7):
    print(kth_last(n1, i).val)
