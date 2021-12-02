import itertools
import random
import timeit
from typing import List, Callable, Optional, Tuple

from decorators import empty, solution_only, template_only

# A HashFunction takes an input string (str) and a capacity (int), and produces a hash value (int).
HashFunction = Callable[[str, int], int]


# A hash table class with a set capacity that can use a custom hashing function.
class HashTable:
    capacity: int
    table: List[List[Tuple[str, int]]]
    hash_function: HashFunction

    def __init__(self, capacity: int, hash_function: HashFunction):
        self.capacity = capacity
        self.table = [[] for _ in range(capacity)]
        self.hash_function = hash_function

    def put(self, key: str, value: int) -> None:
        hash_value = self.hash_function(key, self.capacity)
        bucket = self.table[hash_value]
        entry = [(k, v) for k, v in bucket if k == key]
        if len(entry) > 0:
            bucket.remove(entry[0])
        bucket.append((key, value))

    def get(self, key: str) -> Optional[int]:
        hash_value = self.hash_function(key, self.capacity)
        bucket = self.table[hash_value]
        entry = [(k, v) for k, v in bucket if k == key]
        if entry is []:
            return None
        else:
            return entry[0][1]


@empty
# This function calculates the hash of a string, using Python's `hash` function.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_python_hash(string: str, s: int) -> int:
    return abs(hash(string)) % s


@empty
# This function calculates the hash of a string, returning the length of a string.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_string_length(string: str, s: int) -> int:
    return len(string) % s


@empty
# This function calculates the hash of a string, always returning the same, predetermined, ultimate answer.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_ultimate(string: str, s: int) -> int:
    return 42 % s


# This function generates a heck load of strings.
# You can tweak the string length and the number of strings that will be generated.
def generate_a_heck_load_of_strings(str_len: int, num_strings: int) -> List[str]:
    # Fix random seed for repeatability
    random.seed(421337)
    return [
        # Choose between numbers [48,58], capital letters [65,91], and small letters [97,123].
        "".join(chr(random.choice(list(itertools.chain(range(48, 58), range(65, 91), range(97, 123)))))
                # Choose a random string length between 1 and str_len, inclusive
                for _ in range(random.randint(1, str_len)))
        # Do this num_strings times
        for _ in range(num_strings)
    ]


# Builds an experiment with one of the tables.
def experiment_builder(strings: List[str], table: HashTable) -> Callable[[], None]:
    @empty
    # The `experiment` function is the one being timed by `timeit`.
    # In this function, you can write the code that should be executed during the experiment.
    # You can use hash table `table` and the list of strings `List[str]`.
    def experiment():
        for s in strings:
            table.put(s, 42)
        for s in strings:
            table.get(s)

    return experiment


# In this function, the experiment is performed.
def main():
    if solution_only:
        num_repeats = 10
        capacities = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
        str_len = 100
    if template_only:
        num_repeats = ...  # TODO: choose a number of repeats for each (hash function, capacity) combination
        capacities = [10, 20, ...]  # TODO: choose appropriate hash table capacities to run the experiment for
        str_len = ...  # TODO: choose the maximum length of the generated strings

    print("Size\tPython\tlen(str)\tultimate")
    for capacity in capacities:
        if solution_only:
            num_strings = capacity * 2
        if template_only:
            num_strings = capacity * ...  # TODO: choose the number of strings to generate, based on the capacity
        strings = generate_a_heck_load_of_strings(str_len, num_strings)
        table1 = HashTable(capacity, hash_python_hash)
        table2 = HashTable(capacity, hash_string_length)
        table3 = HashTable(capacity, hash_ultimate)
        time1 = timeit.Timer(experiment_builder(strings, table1)).timeit(num_repeats)
        time2 = timeit.Timer(experiment_builder(strings, table2)).timeit(num_repeats)
        time3 = timeit.Timer(experiment_builder(strings, table3)).timeit(num_repeats)
        print(f"{capacity}\t{time1}\t{time2}\t{time3}")


if __name__ == '__main__':
    main()
