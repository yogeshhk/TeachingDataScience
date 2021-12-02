from typing import List

from decorators import empty, remove


@empty
# Sorts a list of Dutch mobile phone numbers in non-decreasing using LSD radix sort.
def radix_sort_lsd(phone_numbers: List[str]) -> List[str]:
    for i in range(9, 1, -1):  # Iterates over the positions 9, 8, 7, ..., 2. The "06" part never changes.
        phone_numbers = sort_by_char(phone_numbers, i)
    return phone_numbers


@remove
def sort_by_char(phone_numbers: List[str], pos: int) -> List[str]:
    buckets: List[List[str]] = [[] for _ in range(10)]
    for phone_number in phone_numbers:
        bucket = int(phone_number[pos])
        buckets[bucket] += [phone_number]
    res: List[str] = []
    for bucket in buckets:
        res += bucket
    return res
