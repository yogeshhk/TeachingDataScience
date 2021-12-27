#   Created by Elshad Karimov on 20/04/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

#  Update / add an element to the dictionary

myDict = {'name': 'Edy', 'age': 26}
myDict['address'] = 'London'
print(myDict)

#  Traverse through a dictionary

def traverseDict(dict):
    for key in dict:
        print(key, dict[key])

traverseDict(myDict)

#  Searching a dictionary


def searchDict(dict, value):
    for key in dict:
        if dict[key] == value:
            return key, value
    return 'The value does not exist'
print(searchDict(myDict, 27))

#  Delete or remove a dictionary

myDict.pop('name')




# sorted method
myDict = {'eooooa': 1, 'aas': 2, 'udd': 3, 'sseo': 4, 'werwi': 5}

print(sorted(myDict, key=len))
