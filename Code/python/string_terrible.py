def terrible() -> str:
    result = ''
    for c in my_long_string:
        if c.isnumeric():
            result += c
    return result
