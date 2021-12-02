from decorators import empty


class Node:

    def __init__(self, value: str):
        self.value = value
        self.next_node = None


class SLL:

    def __init__(self):
        self.head = None
        self.size = 0

    @empty
    # This is the function, that you have to implement.
    # It reverses the singly linked list
    def reverse(self):
        current_pointer = self.head
        next_pointer = None
        prev_pointer = None
        while current_pointer is not None:
            next_pointer = current_pointer.next_node
            current_pointer.next_node = prev_pointer
            prev_pointer = current_pointer
            current_pointer = next_pointer
        self.head = prev_pointer

    # Adds a new element to the list as the first element.
    def add_first(self, value: str):
        new_node = Node(value)
        new_node.next_node = self.head
        self.head = new_node
        self.size += 1

    # Removes the first element in the list.
    def remove_first(self) -> str:
        # No elements in the list
        if self.head is None:
            return ""
        temp = self.head
        self.head = self.head.next_node
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
        current_node = self.head
        while current_node.next_node is not None:
            current_node = current_node.next_node
        # Append new element to the end
        current_node.next_node = new_node
        self.size += 1

    # Removes the last element in the list.
    def remove_last(self) -> str:
        # No elements in the list
        if self.head is None:
            return ""
        current_node = self.head
        # One element in the list
        if current_node.next_node is None:
            return self.remove_first()
        # Find last element
        while current_node.next_node.next_node is not None:
            current_node = current_node.next_node
        temp = current_node.next_node
        current_node.next_node = None
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
        new_node.next_node = current_node.next_node
        current_node.next_node = new_node
        self.size += 1

    # Remove an element at a certain index
    # Does nothing if there is no element at the specified index
    def remove_at_position(self, idx: int) -> str:
        if self.head is None:
            return ""

        if idx == 0:
            return self.remove_first()

        if idx == self.size - 1:
            return self.remove_last()

        if idx > self.size - 1:
            return ""

        current_node = self.head

        while idx > 1:
            current_node = current_node.next_node
            idx -= 1
        temp = current_node.next_node
        current_node.next_node = current_node.next_node.next_node
        self.size -= 1
        return temp.value

    # Returns the value of the first node,
    # None if the head is None
    def get_head(self) -> str:
        if self.head is None:
            return ""
        return self.head.value
