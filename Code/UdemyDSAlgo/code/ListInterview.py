#   Created by Elshad Karimov on 15/04/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

#  Question 1

mylist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]


def findMissing(list, n):
    sum1 = sum(list)
    sum2 = 100*101/2
    print(sum2-sum1)


# findMissing(mylist, 100)

# Question 2
def findPairs(list, sum):
    for i in range(len(list)):
        for j in range(i+1,len(list)):
            if (list[i]+list[j]) == sum:
                print(list[i],list[j])
# findPairs(mylist, 100)


# Question 3
import numpy as np 
myArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

def findNumber(array, number):
    for i in range(len(array)):
        if array[i] == number:
            print(i)

findNumber(myArray, 12)

# Question 3

def findMaxProduct(array):
    maxProduct = 0
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            if array[i]*array[j] > maxProduct:
                maxProduct = array[i]*array[j]
                pairs = str(array[i])+ "," + str(array[j])
    print(pairs)
    print(maxProduct)

findMaxProduct(myArray)



#Question 5 - isqunieuq
def isUnique(list):
  a=[]
  for i in list:
    if i in a:
        print(i)
        return False
    else:
        a.append(i)
  return True

print(isUnique(myList))



#Question 6 - permutation

def permuntation(list1, list2):
    print(list1)
    print(list2.reverse())
    if list1 == list2:   # if list1 == list2.reverse() -- false
        return True
    else:
        return False

# print(permuntation([1,2,3], [3,2,1]))


# Question 7

def rotate_matrix(matrix):
    '''rotates a matrix 90 degrees clockwise'''
    n = len(matrix)
    for layer in range(n // 2):
        first, last = layer, n - layer - 1
        for i in range(first, last):
            # save top
            top = matrix[layer][i]

            # left -> top
            matrix[layer][i] = matrix[-i - 1][layer]

            # bottom -> left
            matrix[-i - 1][layer] = matrix[-layer - 1][-i - 1]

            # right -> bottom
            matrix[-layer - 1][-i - 1] = matrix[i][- layer - 1]

            # top -> right
            matrix[i][- layer - 1] = top
    return matrix

matrix = [[1,2], [3,4]]
print(matrix)
print(rotate_matrix(matrix))