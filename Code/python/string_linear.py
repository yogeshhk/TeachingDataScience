def linear() -> str:
    result = list()
    for c in my_long_string:
        if c.isnumeric():
            result.append(c)
    return ''.join(result)
