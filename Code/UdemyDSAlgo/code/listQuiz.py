#   Created by Elshad Karimov on 23/04/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

# Q-1. What will be the output of the following code block?
a=[1,2,3,4,5,6,7,8,9]
print(a[::2])

# Q-2. What will be the output of the following code snippet?

a=[1,2,3,4,5,6,7,8,9]
a[::2]=10,20,30,40,50,60
print(a)
# A. ValueError: attempt to assign sequence of size 6 to extended slice of size 5
# B. [10, 2, 20, 4, 30, 6, 40, 8, 50, 60]
# C. [1, 2, 10, 20, 30, 40, 50, 60]
# D. [1, 10, 3, 20, 5, 30, 7, 40, 9, 50, 60]


 
# Q-3. What will be the output of the following code snippet?

a=[1,2,3,4,5]
print(a[3:0:-1])
# A. Syntax error
# B. [4, 3, 2]
# C. [4, 3]
# D. [4, 3, 2, 1]


 

# Q-4. What will be the output of the following code snippet?

def f(value, values):
    v = 1
    values[0] = 44
t = 3
v = [1, 2, 3]
f(t, v)
print(t, v[0])
# A. 1 44
# B. 3 1
# C. 3 44
# D. 1 1


 

# Q-5. What is the correct command to shuffle the following list?

fruit=['apple', 'banana', 'papaya', 'cherry']
# A. fruit.shuffle()
# B. shuffle(fruit)
# C. random.shuffle(fruit)
# D. random.shuffleList(fruit)


 


 
# Q-6. What will be the output of the following code snippet?

data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
def fun(m):
    v = m[0][0]

    for row in m:
        for element in row:
            if v < element: v = element

    return v
print(fun(data[0]))
# A. 1
# B. 2
# C. 3 
# D. 4
# E. 5
# F. 6


 

# Q-7. What will be the output of the following code snippet?

arr = [[1, 2, 3, 4],
       [4, 5, 6, 7],
       [8, 9, 10, 11],
       [12, 13, 14, 15]]
for i in range(0, 4):
    print(arr[i].pop())
# A. 1 2 3 4
# B. 1 4 8 12
# C. 4 7 11 15 
# D. 12,13,14,15


 

# Q-8. What will be the output of the following code snippet?

def f(i, values = []):
    values.append(i)
    print (values)
    return values
f(1)
f(2)
f(3)
# A. [1] [2] [3]
# B. [1, 2, 3]
# C. [1] [1, 2] [1, 2, 3]
# D. 1 2 3


 

# Q-9. What will be the output of the following code snippet?

arr = [1, 2, 3, 4, 5, 6]
for i in range(1, 6):
    arr[i - 1] = arr[i]
for i in range(0, 6): 
    print(arr[i], end = " ")
# A. 1 2 3 4 5 6
# B. 2 3 4 5 6 1
# C. 1 1 2 3 4 5 
# D. 2 3 4 5 6 6


 

# Q-10. What will be the output of the following code snippet?

fruit_list1 = ['Apple', 'Berry', 'Cherry', 'Papaya']
fruit_list2 = fruit_list1
fruit_list3 = fruit_list1[:]

fruit_list2[0] = 'Guava'
fruit_list3[1] = 'Kiwi'

sum = 0
for ls in (fruit_list1, fruit_list2, fruit_list3):
    if ls[0] == 'Guava':
        sum += 1
    if ls[1] == 'Kiwi':
        sum += 20

print(sum)
# A. 22
# B. 21
# C. 0
# D. 43

