# Find n_C_k

def combination(n, k):
    results = []

    def find_path(start_number, current_path):
        if len(current_path) == k:
            results.append(current_path.copy())
            return
        for i in range(start_number, n + 1):
            current_path.append(i)
            find_path(i + 1, current_path)
            current_path.pop()

    find_path(1, [])
    return results


results = combination(4,2)
print(results)
