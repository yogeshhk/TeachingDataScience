
# def solution(A):
#     # write your code in Python 3.6
#     been_there = set()
#     resultant_tuples_list = []
#     for i in range(len(A)):
#         been_there.add(A[i])
#         for j in range(i+1,len(A)):
#             if A[j] in been_there:
#                 continue
#             a = A[i]
#             b = A[j]
# #             been_there.add(A[j])
#             for num in range(a+1,b):
#                 if num in A:
#                     break
#                 resultant_tuples_list.append((i,j))
#             else:
#                 resultant_tuples_list.append((i,j))
#     if len(resultant_tuples_list) == 0:
#         return -1
#     
#     min_distance = 100000000
#     for pairs in resultant_tuples_list:
#         distance = abs(A[pairs[0]]-A[pairs[1]])
#         if distance < min_distance:
#             min_distance = distance
#     if min_distance > 1000000:
#         return -1
#     return min_distance
    
#             
#                 
# A = [0,3,3,7,5,3,11,1]

# A = [7,3,7,3,1,3,4,1]
# 
# def contains_all_destinations(sublist,destinations):
#     if len(sublist)<len(destinations):
#         return False
#     for d in destinations:
#         if d not in sublist:
#             return False
#     return True
# 
# def solution(A):
#     min_distance = 1000000000
#     destinations = set(A)
#     for i in range(len(A)):
#         a = A[i]    
#         sublist = [a]    
#         for j in range(i+1,len(A)):
#             sublist.append(A[j])
#             if contains_all_destinations(sublist,destinations):
#                 distance = j - i + 1
#                 if distance < min_distance:
#                     min_distance = distance
#     return min_distance
                

def get_prefixes(S):
    prefixes_list = []
    for i in range(len(S)+1):
        prefixes_list.append(S[:i])
    return prefixes_list

def get_suffixes(S):
    suffixes_list = []
    for i in range(len(S)+1):
        suffixes_list.append(S[i:])
    return suffixes_list

def solution(S):
    prefixes_list = get_prefixes(S)
    suffixes_list = get_suffixes(S)
    intersection = list(set(prefixes_list).intersection(set(suffixes_list)))
    intersection.remove(S)
    max_length = -1
    for word in intersection:
        if len(word) > max_length:
            max_length = len(word)
    return max_length
    
S = "codility"
result = solution(S)
print(result)