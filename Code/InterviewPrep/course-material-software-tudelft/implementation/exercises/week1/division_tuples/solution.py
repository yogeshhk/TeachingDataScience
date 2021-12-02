from decorators import empty


@empty
# Divides x by y and returns the quotient and remainder
def divide(x: int, y: int) -> (int, int):
    return x // y, x % y
