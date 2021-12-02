import decorators


@decorators.empty
# This function calculates the hash of a string, using a simple hasing function.
# The hash value should lie between 0 (inclusive) and s (exclusive).
def simple_hash(string: str, s: int) -> int:
    a = 1003
    m = 1_000_000_009
    b = 0
    for char in string:
        b = (ord(char) + b * a) % m
    return b % s
