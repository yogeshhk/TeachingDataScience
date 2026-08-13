# Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
#
# Each number in candidates may only be used once in the combination.
#
# Note: The solution set must not contain duplicate combinations.
#
#
#
# Example 1:
#
# Input: candidates = [10,1,2,7,6,1,5], target = 8
# Output:
# [
# [1,1,6],
# [1,2,5],
# [1,7],
# [2,6]
# ]

# KEY: Decision tree, to include each digit or not, so O(2^n)
# Sort. To avoid duplicates, travers by skipping duplicate digits

def combination_sum(numbers, target):
    numbers.sort()
    results = []

    def find_path(current_path, start_position, remaining_target):
        if remaining_target == 0:
            results.append(current_path.copy())
        if remaining_target < 0:
            return

        previous_number = -1
        for i in range(start_position,len(numbers)):
            if numbers[i] == previous_number:
                continue
            current_path.append(numbers[i])
            find_path(current_path, i + 1, remaining_target - numbers[i])
            current_path.pop()

            previous_number = numbers[i]

    find_path([], 0, target)
    return results


results = combination_sum([10, 1, 2, 7, 6, 1, 5], 8)
print(results)
