In this exercise you have to implement two functions:
    
- `first_index_of`
- `all_indices_of`
 
The function `first_index_of` takes two things as input, a number `num` and a list `l`, and returns the first index of the number `n` in list `l`.

However as you might have noticed from the name of the function, we now only return the first index in the list. If we are given the list, `[1, 2, 3, 1]`, we won't find the second instance of the number `1` in our list. To solve this, we will implement the function `all_indices_of`. 

The function `all_indices_of` takes the same type of input as `first_index_of`, however it now returns a generator with all the indices of the number `n` in list `l`. 

<details>
    <summary> Generator </summary>
    
    A generator in python is used to create an iterator, this allows us to get the values from a function one by one.
    
    For example a generator that returns all numbers from 0 to 4 would look like this:
    
    <pre>
    def example_generator():
        for x in range(0, 5):
            yield x
    </pre>
    
    More information on generators can be found on <a href="https://www.geeksforgeeks.org/python-list-comprehensions-vs-generator-expressions/">this link</a>.
</details>
