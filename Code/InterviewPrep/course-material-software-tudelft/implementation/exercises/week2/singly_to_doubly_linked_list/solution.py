class Node:

    def __init__(self, value: str):
        self.value = value
        self.next_node = None
        self.prev_node = None


class LL:

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    # Adds a new element to the list as the first element.
    def add_first(self, value: str):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            self.size += 1
            return
        new_node.next_node = self.head
        self.head.prev_node = new_node
        self.head = new_node
        self.size += 1

    # Removes the first element in the list.
    def remove_first(self) -> str:
        # No elements in the list
        if self.head is None:
            return None
        if self.size == 1:
            temp = self.head
            self.head = None
            self.tail = None
            self.size -= 1
            return temp.value
        temp = self.head
        self.head = self.head.next_node
        self.head.prev_node = None
        self.size -= 1
        return temp.value

    # Add a new element to the end of the list.
    def add_last(self, value: str):
        # No elements in the list
        if self.head is None:
            self.add_first(value)
            return
        # Find last element in the list
        new_node = Node(value)
        new_node.prev_node = self.tail
        self.tail.next_node = new_node
        self.tail = new_node
        self.size += 1

    # Removes the last element in the list.
    def remove_last(self) -> str:
        # No elements in the list
        if self.head is None:
            return None
        current_node = self.head
        # One element in the list
        if current_node.next_node is None:
            return self.remove_first()
        # Find last element
        temp = self.tail
        self.tail = self.tail.prev_node
        self.tail.next_node = None
        self.size -= 1
        return temp.value

    # Add a new element at a certain index
    # If the index is larger than the size of the list this will append it to the end of the list
    def add_at_position(self, idx: int, value: str):
        if idx == 0 or self.head is None:
            self.add_first(value)
            return

        if idx > self.size - 1:
            self.add_last(value)
            return

        new_node = Node(value)
        current_node = self.head

        while idx > 1:
            current_node = current_node.next_node
            idx -= 1
        new_node.prev_node = current_node
        new_node.next_node = current_node.next_node
        new_node.next_node.prev_node = new_node
        current_node.next_node = new_node
        self.size += 1

    # Remove an element at a certain index
    # Does nothing if there is no element at the specified index
    def remove_at_position(self, idx: int) -> str:
        if self.head is None:
            return None

        if idx == 0:
            return self.remove_first()

        if idx == self.size - 1:
            return self.remove_last()

        if idx > self.size - 1:
            return None

        current_node = self.head

        while idx > 1:
            current_node = current_node.next_node
            idx -= 1
        temp = current_node.next_node
        current_node.next_node = current_node.next_node.next_node
        current_node.next_node.prev_node = current_node
        self.size -= 1
        return temp.value

    # Returns the value of the first node,
    # None if the head is None
    def get_head(self) -> str:
        if self.head is None:
            return None
        return self.head.value

    # Returns the value of the first node,
    # None if the head is None
    def get_tail(self) -> str:
        if self.tail is None:
            return None
        return self.tail.value
