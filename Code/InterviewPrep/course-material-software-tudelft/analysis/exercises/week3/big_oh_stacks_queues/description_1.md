For the code snippet below, work out the worst-case run time complexity in terms of big-Oh notation.

```python

def method_x(s: Stack[int], item: int):
    temp = Stack()
    for i in range(len(s)):
        temp.push(s.pop())
    s.push(item)
    for i in range(len(temp)):
        s.push(temp.pop())
```
1) Derive the run time equation of the code and explain all the terms in your equations.

2) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but full 
proof is not needed.​

