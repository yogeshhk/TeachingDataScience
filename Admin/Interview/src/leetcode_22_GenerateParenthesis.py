# Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
#
#
#
# Example 1:
#
# Input: n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]
# Example 2:
#
# Input: n = 1
# Output: ["()"]

# if n = 3, you have 3 open and 3 close parenthesis at max
# You can add any number of OPEN parenthesis but while adding CLOSE parenthesis, need to check if CLOSE < OPEN

def generate_paranthesis(n):
    stack = []
    result = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            result.append("".join(stack))

        if openN < n:
            stack.append("(")
            backtrack(openN+1, closedN)
            stack.pop()

        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN+1)
            stack.pop()

    backtrack(0,0)

    return result

result = generate_paranthesis(3)
print(result)

