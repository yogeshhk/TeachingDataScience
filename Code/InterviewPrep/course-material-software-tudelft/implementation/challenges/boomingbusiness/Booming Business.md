You are an expert in bonsai, the Japanese art of cultivating small trees in small containers.
Every year, you win the Bonsai Association's Pruning Competition (BAPC).
With all this talent, it would be a shame not to turn your hobby into your job.
Recently, you have rented a small store where you will sell your creations.
Now you need to make a window display to draw in customers.
Of course, you would like to grow the most impressive tree that will fit the window, but the window is only so tall, and the floor of the display can only bear so much weight.
Therefore, you want a tree that is exactly so tall and so heavy that it can fit in your window.

Being an expert, you know that by definition a bonsai tree consists of a single branch, with 0 or more smaller bonsai trees branching off from that branch.

![https://i.gyazo.com/67c1020efff4405c9f6bbeb868904b7b.png](https://i.gyazo.com/67c1020efff4405c9f6bbeb868904b7b.png)
<br>_Figure 1: Four distinct examples of bonsai trees._

The height and weight of a bonsai tree can be carefully determined.
A tree's weight is equal to the number of branches that appear in it.
The weights of the trees in Figure 1 are 1, 4, 6 and 6, respectively.
A tree's height is equal to the length of the longest chain of branches from the root to the top of the tree.
The heights of the trees in Figure 1 are 1, 2, 3 and 3, respectively.

To make the most use of your window, you want to produce a bonsai tree of the precise height and weight that it can support.
To get an idea of the number of options available to you, you would like to know how many different trees you could possibly grow for your store.
Given a height and a weight, can you determine the number of trees with exactly that height and weight?
Because the number may be very large, you may give your answer modulo 1,000,000,007.

#### Input
A single line containing two integers, \\(h\\) and \\(w\\), with \\(1 \leq h, w \leq 300\\).

#### Output
Output a single line containing a single integer, the number of bonsai trees of height \\(h\\) and weight \\(w\\), modulo \\(10^9 + 7\\).

#### Examples
For each example, the first block is the input and the second block is the corresponding output.

##### Example 1
```
2 4
```
```
1
```
##### Example 2
```
3 5
```
```
7
```
##### Example 3
```
20 50
```
```
573689752
```

Source: BAPC 2017
