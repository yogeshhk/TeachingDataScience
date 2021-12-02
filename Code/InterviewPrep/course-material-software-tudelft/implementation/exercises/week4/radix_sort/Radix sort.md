In this assignment, you are expected to implement the LSD (least-significant-digit-first) radix sort algorithm to sort a sequence of Dutch mobile phone numbers in non-decreasing order.

You may assume that every string in the input list starts with `06` and is followed by eight digits.

Your algorithm needs to run in \\(\mathcal O(n)\\) time, where \\(n\\) is the number of items to sort.

The LSD radix sort algorithm works as follows:

1. For every digit (starting at the least significant digit):
    1. Create buckets for the possible digits.
    2. Add phone numbers to the bucket corresponding to the current digit position.
    3. Add all buckets in one resulting list, and start the next iteration with this list.
