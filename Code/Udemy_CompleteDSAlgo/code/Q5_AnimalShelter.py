#   Created by Elshad Karimov on 02/06/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

# Implement a cat and dog queue for an animal shelter.

class AnimalShelter():
  def __init__(self):
    self.cats = []
    self.dogs = []
  
  def enqueue(self, animal, type):
    if type == 'Cat':
      self.cats.append(animal)
    else:
      self.dogs.append(animal)
    
  def dequeueCat(self):
    if len(self.cats) == 0:
      return None
    else:
      cat = self.cats.pop(0)
      return cat
  
  def dequeueDog(self):
    if len(self.dogs) == 0:
      return None
    else:
      dog = self.dogs.pop(0)
      return dog
  
  def dequeueAny(self):
    if len(self.cats) == 0:
      result = self.dogs.pop(0)
    else:
      result = self.cats.pop(0)
    return result

customQueue = AnimalShelter()
customQueue.enqueue('Cat1', 'Cat')
customQueue.enqueue('Cat2', 'Cat')
customQueue.enqueue('Dog1', 'Dog')
customQueue.enqueue('Cat3', 'Cat')
customQueue.enqueue('Dog2', 'Dog')
print(customQueue.dequeueAny())
