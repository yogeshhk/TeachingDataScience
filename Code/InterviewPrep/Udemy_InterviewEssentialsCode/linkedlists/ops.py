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

# Get the nth element in the list
def get_nth(head, n):
    count = 0
    cur = head
    while (cur):
        if (count == n):
            return cur.val
        count += 1
        cur = cur.next

# Insert and return the new head
def insert_beginning(head, new_val):
    new_node = Node(new_val)
    new_node.next = head
    return new_node

# Delete and return the new head
def delete_beginning(head):
    return head.next

# Insert at an index (assume valid index)
def insert_at(head, new_val, index):
  if index == 0:
    return insert_beginning(head, new_val)
  count = 0
  cur = head
  while (cur):
      if (count == index - 1):
        new_node = Node(new_val)
        new_node.next = cur.next
        cur.next = new_node
        return head
      count += 1
      cur = cur.next

# Delete at an index (assume valid index)
def delete_at(head, index):
    if index == 0:
      return delete_beginning(head)
    count = 0
    cur = head
    while (cur):
        if (count == index - 1):
          cur.next = cur.next.next
          return head
        count += 1
        cur = cur.next

n1 = Node(3)
n2 = Node(1)
n3 = Node(5)
n4 = Node(2)
n1.next = n2
n2.next = n3
n3.next = n4

head = n1
print("Initial list:")
print_list(head)

print("Indexing: ")
for i in range(4):
    print("Index " + str(i) + ": " + str(get_nth(head, i)))

head = insert_beginning(head, 1)
head = insert_beginning(head, 2)
print("List after inserting 1, 2:")
print_list(head)

head = delete_beginning(head)
head = delete_beginning(head)
print("List after deleting 1, 2:")
print_list(head)

head = delete_at(head, 2)
print("List after deleting 3rd element (index 2)")
print_list(head)

head = insert_at(head, 5, 2)
print("List after inserting to 3rd position (index 2)")
print_list(head)