# Given two linked lists representing 2 numbers, digits are stored in reverse order. Add both and return result as list
# Example: (2->4->3) + (5->6->4), so numbers are 342 + 465 = 807, so result is (7->0->8)
# Ref: https://www.youtube.com/watch?v=sUicrnHwA0s&list=PLiC1doDIe9rDFw1v-pPMBYvD6k1ZotNRO&index=2

# Definition of singly-linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# # make numbers from lists
# # add them
# # deconstruct list from result
# def make_num(l):
#     result = 0
#     for i in l:
#         result += i * 10**i
#     return result
#
# def make_list(num):
#     while num:
#         digit = num % 10
#         l = ListNode(digit)
#         num //= 10
#
#     return l
#
# def addTwoNumbers(l1,l2):
#     num1 = make_num(l1)
#     num2 = make_num(l2)
#     result = num1 + num2

# position wise addition, and carry forward if necessary
def addTwoNumbers(l1, l2):
    added = ListNode(val=(l1.val + l2.val) % 10)
    carry_over = (l1.val + l2.val) // 10
    current_node = added
    while (l1.next and l2.next):
        l1 = l1.next
        l2 = l2.next
        current_node.next = ListNode(val=(carry_over + l1.val + l2.val) % 10)
        carry_over = (carry_over + l1.val + l2.val) // 10
        current_node = current_node.next

    while(l1.next):
        l1 = l1.next
        current_node.next = ListNode(val=(carry_over + l1.val) % 10)
        carry_over = (carry_over + l1.val ) // 10
        current_node = current_node.next

    while(l2.next):
        l2 = l2.next
        current_node.next = ListNode(val=(carry_over + l2.val) % 10)
        carry_over = (carry_over + l2.val ) // 10
        current_node = current_node.next

    if carry_over > 0:
        current_node.next = ListNode(val=1)
    return added