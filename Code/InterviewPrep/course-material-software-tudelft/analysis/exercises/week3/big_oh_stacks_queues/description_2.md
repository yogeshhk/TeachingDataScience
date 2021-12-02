For the code snippet below, work out the worst-case run time complexity in terms of big-Oh notation.

```python

def method_y(q: Queue[int], item: int):
    temp = Queue()
    flag = False
    for i in range(len(q)):
        current = q.dequeue()
        if current == item:
            flag = True
            break
        else:
            temp.enqueue()
    for i in range(len(temp)):
        q.enqueue(temp.dequeue())
    return flag
```

1) Derive the run time equation of the code and explain all the terms in your equations.

2) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but full 
proof is not needed.​