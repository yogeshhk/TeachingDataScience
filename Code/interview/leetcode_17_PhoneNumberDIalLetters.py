# Construct Mobile Letters combinations for a given phone number
# Example: 23 = ["ad","ae","af","bd","be","bf","cd","ce","cf"]

def constructCombinations(digits):
    phone_map = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    if digits == "":
        return []

    numbers = list(phone_map[digits[0]])
    for digit in digits[1:]:
        numbers = [old + new for old in numbers for new in list(phone_map[digit])]
    return numbers
