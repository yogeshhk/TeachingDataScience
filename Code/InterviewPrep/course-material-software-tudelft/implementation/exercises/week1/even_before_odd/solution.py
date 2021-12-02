from typing import List
from decorators import empty, remove


@empty
# Recursively places all even numbers before all odd numbers
def even_before_odd(xs: List[int]):
    helper(xs, 0, len(xs))


@remove
# Places even before odd in xs[low:high]
def helper(xs: List[int], low: int, high: int):
    if low >= high:
        return

    # If odd
    if xs[low] % 2:
        # Move value to the last unchecked position
        xs[low], xs[high - 1] = xs[high - 1], xs[low]

        # Alternatively, we can move the value to the end:
        # xs.append(xs.pop(low))
        # This is O(n) due to the pop operation (which causes all elements to be shifted),
        # whereas a swap takes constant time.

        # Recursive call, decrease high as there's one more item in the back we don't need to look at
        helper(xs, low, high - 1)

    # If even
    else:
        # Recursive call, increase low to skip the even value and leave it at the front
        helper(xs, low + 1, high)
