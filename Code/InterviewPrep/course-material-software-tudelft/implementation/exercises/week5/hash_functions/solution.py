from decorators import empty


@empty
# This function calculates the hash of a string, using a variant on the GNU-cc1 hashing algorithm.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_gnu_cc1(string: str, s: int) -> int:
    b = len(string)
    for char in string:
        b = ord(char) ^ b * 33
    return b % s


@empty
# This function calculates the hash of a string, using a variant on the GNU-cpp hashing algorithm.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_gnu_cpp(string: str, s: int) -> int:
    b = 0
    for char in string:
        b = ord(char) + b * 4
    return b % s


@empty
# This function calculates the hash of a string, using Python's `hash` function.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def hash_python_hash(string: str, s: int) -> int:
    return abs(hash(string)) % s
