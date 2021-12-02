For NWERC 2018, the organisers have done something rather special with the balloons.
Instead of buying balloons of equal size, they bought one balloon of every integer size from \\(1\\) up to \\(n\\). A balloon of size \\(s\\) has a capacity of \\(s\\) decilitres.

To avoid inflating the balloons by hand, the organisers also bought \\(n\\) helium gas canisters. Each canister can only be used to inflate one balloon, and must be emptied completely into that balloon
(it is not possible to disconnect a canister from a balloon before the canister has been fully used).

Unfortunately the gas canisters were bought at a garage sale, and may contain differing amounts of helium. Some may even be empty! To make the best of this challenging situation, the canisters will have to be paired with the balloons smartly.

The organisers want to assign all of the gas canisters to separate balloons, such that the balloon that is inflated the least (relative to its capacity) still contains the maximum possible fraction of helium inside.
What is the maximum such (minimum) fraction that is possible?

Balloons filled beyond their capacity will explode. Explosions are upsetting and must be avoided.

#### Input
The input consists of:

 - One line with the integer \\(n\\) (\\(1 \le n \le 2\cdot10^5\\)), the number of balloons and gas canisters.
 - One line with \\(n\\) integers \\(c_1, \ldots, c_n\\) (\\(0 \le c_i \le n\\) for each \\(i\\)), the amounts of helium in the gas canisters, in decilitres.

#### Output

If it is possible to fill all the balloons without any exploding, output the maximum fraction \\(f\\) such that every balloon can be filled to at least \\(f\\) of its capacity.  Otherwise, output "`impossible`".

Your answer should have an absolute or relative error of at most \\(10^{-6}\\).
For each example, the first block is the input and the second block is the corresponding output.
##### Example 1
```
6
6 1 3 2 2 3
```
```
0.6
```
##### Example 2
```
2
2 2
```
```
impossible
```
##### Example 3
```
5
4 0 2 1 2
```
```
0
```

Source: NWERC 2018
