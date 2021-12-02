#### Setup:
- Each experiment is run 10 times.
- Capacities between `10` and `10240` have been used, everytime multiplying by two.
- The strings have a maximum length of 100.
- For each capacity \\(c\\), \\(2c\\) strings are generated.
- The experiment that is being timed, calls `put` on the hash table for all strings, followed by a call to `get` on the hash table for all strings.

#### Table and Graph
Figure 1 shows the running time for the three hash tables for the different sizes.

![chart](https://i.gyazo.com/18d57c7b6f3a3e58b3fdf3569fb59310.png)
_Figure 1: Runtimes of the different hashing functions. Logarithmic scale. Trend lines generated using a "power" formula (\\(f(x) = ax^b\\)). Graph created in LibreOffice Calc._

#### Observations:
- The python hash function makes the hash table operate in almost linear time, i.e. constant time for each operation.
    - The entries are mapped to different buckets in most cases.
    The sizes of the buckets will be 0.5 on average, so the operations take roughly constant time.
- The "ultimate hash function" makes the hash table operate in almost quadratic time, i.e. linear time for each operation.
    - This happens because all entries are mapped to the same bucket in the hash table, making any operation boil down to an \\(\mathcal O(n)\\) operation on a Python list.
- The hash function that takes the string length makes the hash table operate in almost linear time for sizes < 1000, but becomes more quadratic for larger sizes.
    - Note that for this hash function, only 100 buckets will ever be used.
      This means that for less than 1000 entries, each of the used buckets will have an average size of less than 10, so roughly "constant".
      With more than 1000 entries, the size of each bucket becomes linearly dependent on the amount of entries, which means that (just like with the "ultimate hash function") each operation will take \\(\mathcal O(n)\\) time.
      Still, in this case, the running time is roughly 100 times faster than the "ultimate hash function".
    - The trend line for this data series has a bad fit, so we can't trust the exponent in the formula.
      We can better see the linear component separated from the quadratic component when we split the data series in two, as done in Figure 2.
      ![split](https://i.gyazo.com/a56a1d3b024c6e176d7081b495a6f2e4.png)
      _Figure 2: Runtimes for the hash function that returns the length of the string. Same software/configuration is used as in Figure 1._
