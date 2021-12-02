from typing import List

import decorators


@decorators.empty
# This function takes as input a number and returns `True`
# if the number is prime or `False` if it is not.
def is_prime(num: int) -> bool:
    if num > 1:
        for i in range(2, num):
            if (num % i) == 0:
                return False
        return True
    else:
        return False


@decorators.empty
# This function takes a `lower` and an `upper` bound of
# an interval (inclusive) and returns a list containing all the prime
# numbers in that interval.
def primes_in_range(lower: int, upper: int) -> List[int]:
    return [num for num in range(lower, upper + 1) if is_prime(num)]


@decorators.empty
# This function takes a `lower` and an `upper` bound of
# an interval (inclusive) and returns the number of prime numbers
# in that interval.
def count_primes(lower: int, upper: int) -> int:
    return len(primes_in_range(lower, upper))
