import functools

def compare_intervals(int1, int2):
    # Compare by start time
    if int1[0] > int2[0]:
        return 1
    elif int1[0] < int2[0]:
        return -1
    return 0

def merge(intervals):
    merged = []
    for interval in sorted(intervals, key=functools.cmp_to_key(compare_intervals)):
        if len(merged) > 0 and interval[0] <= merged[-1][1]:
            # Merge the two by taking the greater end time
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)
    return merged

print(merge([[1,3],[2,6],[8,10],[15,18]]))
