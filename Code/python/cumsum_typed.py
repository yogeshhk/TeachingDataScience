from typing import List


def fey(my_list: List[int]) -> List[int]:
    squares = list()
    for i in my_list:
        squares.append(i * i)
    return squares
