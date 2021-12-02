import decorators


@decorators.empty
# This function takes degrees in Celsius and returns the
# corresponding degrees in Fahrenheit, rounded to one decimal place.
def celsius_to_fahrenheit(num: float) -> float:
    return round((num * 1.8) + 32, 1)


@decorators.empty
# This function takes in a `lower` and `upper` bound of an interval (inclusive)
# and returns `True` if there is an Armstrong number in that interval
# or `False` if there is none.
def armstrong_number_in_interval(lower: int, upper: int) -> bool:
    for num in range(lower, upper + 1):
        order = len(str(num))
        result = 0
        temp = num
        while temp > 0:
            digit = temp % 10
            result += digit ** order
            temp //= 10

        if num == result:
            return True
    return False
