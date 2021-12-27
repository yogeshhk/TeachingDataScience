#   Created by Elshad Karimov on 17/05/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

# Question 1 - Remove Dups : Write a code to remove duplicates from an unsorted linked list. 


from LinkedList import LinkedList

def removeDups(ll):
    if ll.head is None:
        return
    else:
        currentNode = ll.head
        visited = set([currentNode.value])
        while currentNode.next:
            if currentNode.next.value in visited:
                currentNode.next = currentNode.next.next
            else:
                visited.add(currentNode.next.value)
                currentNode = currentNode.next
        return ll

def removeDups1(ll):
    if ll.head is None:
        return
    
    currentNode = ll.head
    while currentNode:
        runner = currentNode
        while runner.next:
            if runner.next.value == currentNode.value:
                runner.next = runner.next.next
            else:
                runner = runner.next
        currentNode = currentNode.next
    return ll.head



customLL = LinkedList()
customLL.generate(10, 0, 99)
print(customLL)
removeDups1(customLL)
print(customLL)