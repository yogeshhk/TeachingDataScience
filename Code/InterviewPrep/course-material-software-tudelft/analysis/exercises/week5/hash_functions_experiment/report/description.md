In this exercise, you will implement several hash functions for strings and compare their performance when used in hash tables.

### Hash functions
The hash functions should always return a value between 0 (inclusive) and the size of the hash table (exclusive).

You should implement the following string hashing algorithms:

#### Python's `hash` function
For this hash function, you can use Python's built-in `hash` function.
Make sure that the hash value as returned by `hash` is in the correct range!
If the hash value is negative, make it positive by flipping the sign.
(Note that this hash function is the same as the one in the implementation assignments.)

#### The length of the string 
For this hash function, you should return the length of the string (after putting it in range).

#### The answer to the ultimate question of life, the universe and everything
The computer Deep Thought used 7.5 million years to come up with the answer to this question. So why not use it as hash value for everything?
However, don't forget to put the answer in range.

### Experiment
Most of the code for the experiment has already been given.
The experiment is structured as follows:

- For different capacities \\(c\\):
    - Generate a list of strings with length \\(f(c)\\) (with \\(f\\) being some function. Can be \\(f(c) = c\\), can be something else).
    - Create three `HashTable`s with capacity \\(c\\).
      Each hash table uses one of the hash functions that you created.
    - Perform \\(\Theta(s)\\) `put` and `get` operations on each hash table, using the list of strings you generated.
      This step is timed using [the Python module `timeit`](https://docs.python.org/3.7/library/timeit.html#timeit.Timer).

Some important details are left out for you to fill in:

- How often the experiment should be run for each (hash function, capacity) combination
- The capacities of the hash tables to test with
- The maximum length of the strings that are generated
- The number of strings that are generated (dependent on the capacity of the hash table)
- The code that will be timed

### Report
- Please describe in your report briefly how you filled in the missing details.
- Create a table and/or graph that displays the different running times for the three hash tables for different sizes.
- What do you observe?
