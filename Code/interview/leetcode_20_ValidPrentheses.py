# Given a string check if parenthesis are valid
# Example: "()[]{}" is true

def isValidParenthesis_removal(s):
    replace = True

    while replace:
        start_length = len(s)
        for inner in ["{}", "[]", "()"]:
            s = s.replace(inner, "")
        if start_length == len(s):
            replace = False

    return s == ""

def isValidParenthesis_stack(s):
    close_map = {"{":"}","[":"]","(":")"}
    opens = []
    for symbol in s:
        if symbol in close_map.keys():
            opens.append(symbol)
        elif opens == [] or symbol != close_map[opens.pop()]: # closing bracket first or mismatched bracket
            return False
    return opens == []

