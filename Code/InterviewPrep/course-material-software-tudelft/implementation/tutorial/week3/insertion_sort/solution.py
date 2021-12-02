from typing import List

from decorators import empty


@empty
# Sorts the list of candies by using insertion sort.
def sort(cards: List[int]):
    for i in range(1, len(cards)):
        # Keep track of which card to insert in the correct spot now
        temp = cards[i]
        idx = i
        # Swap the current card with all cards that are smaller
        while idx > 0 and cards[idx - 1] < temp:
            cards[idx] = cards[idx - 1]
            idx -= 1
        cards[idx] = temp
