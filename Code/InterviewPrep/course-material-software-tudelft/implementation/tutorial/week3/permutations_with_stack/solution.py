from typing import List, Set, Tuple

from decorators import empty
from .library import Stack


@empty
def permutations(ingredients: List[str]) -> Set[Tuple[str]]:

    stack1 = Stack()
    stack2 = Stack()

    stack2.push(ingredients)
    index = 0

    while index < len(ingredients) - 1:
        ingredients = stack2.pop()
        for i in range(index, len(ingredients)):
            temp_list = ingredients.copy()
            temp_list[index], temp_list[i] = temp_list[i], temp_list[index]
            stack1.push(temp_list)
        if stack2.is_empty():
            index += 1
            while not stack1.is_empty():
                stack2.push(stack1.pop())

    result = set()

    while not stack2.is_empty():
        result.add(tuple(stack2.pop()))

    return result
