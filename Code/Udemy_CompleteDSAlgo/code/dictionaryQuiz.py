#   Created by Elshad Karimov on 26/04/2020.
#   Copyright © 2020 AppMillers. All rights reserved

# Dictionary Interview Questions


# Q-1. What will be the output of the following code snippet?

a = {(1,2):1,(2,3):2}
print(a[1,2])
# A. Key Error
# B.  1
# C. {(2,3):2}
# D. {(1,2):1}
 

# Q-2. What will be the output of the following code snippet?

a = {'a':1,'b':2,'c':3}
# print (a['a','b'])
# A. Key Error
# B. [1,2]
# C. {‘a’:1,’b’:2}
# D. (1,2)

 

# Q-3. What will be the output of the following code block?

fruit = {}

def addone(index):
    if index in fruit:
        fruit[index] += 1
    else:
        fruit[index] = 1
        
addone('Apple')
addone('Banana')
addone('apple')
print (len(fruit))
# A. 1 
# B. 2
# C. 3 
# D. 4

 

# Q-4. What will be the output of the following code block?

arr = {}
arr[1] = 1
arr['1'] = 2
arr[1] += 1

sum = 0
for k in arr:
    sum += arr[k]

print(sum)
# A. 1 
# B. 2
# C. 3 
# D. 4

 

# Q-5. What will be the output of the following code snippet?

my_dict = {}
my_dict[1] = 1
my_dict['1'] = 2
my_dict[1.0] = 4

sum = 0
for k in my_dict:
    sum += my_dict[k]
    
print (sum)
# A. 7
# B. Syntax error
# C. 3
# D. 6

 

# Q-6. What will be the output of the following code snippet?

my_dict = {}
my_dict[(1,2,4)] = 8
my_dict[(4,2,1)] = 10
my_dict[(1,2)] = 12

sum = 0
for k in my_dict:
    sum += my_dict[k]

print (sum)
print(my_dict)
# A. Syntax error
# B. 30   
#     {(1, 2): 12, (4, 2, 1): 10, (1, 2, 4): 8}
# C. 47
#     {(1, 2): 12, (4, 2, 1): 10, (1, 2, 4): 8}
# D. 30
#     {[1, 2]: 12, [4, 2, 1]: 10, [1, 2, 4]: 8}

 

# Q-7. What will be the output of the following code snippet?

box = {}
jars = {}
crates = {}
box['biscuit'] = 1
box['cake'] = 3
jars['jam'] = 4
crates['box'] = box
crates['jars'] = jars
# print(len(crates[box]))
# A. 1
# B. 3
# C. 4
# D. Type Error

 

# Q-8. What will be the output of the following code block?

dict = {'c': 97, 'a': 96, 'b': 98}

for _ in sorted(dict):
    print (dict[_])
# A. 96 98 97
# B. 96 97 98
# C. 98 97 96
# D. NameError

 

# Q-9. What will be the output of the following code snippet?

rec = {"Name" : "Python", "Age":"20"}
r = rec.copy()
print(id(r) == id(rec))
# A. True
# B. False
# C. 0
# D. 1

 

# Q-10. What will be the output of the following code snippet?

rec = {"Name" : "Python", "Age":"20", "Addr" : "NJ", "Country" : "USA"}
id1 = id(rec)
del rec
rec = {"Name" : "Python", "Age":"20", "Addr" : "NJ", "Country" : "USA"}
id2 = id(rec)
print(id1 == id2)
# A. True
# B. False
# C. 1
# D. Exception