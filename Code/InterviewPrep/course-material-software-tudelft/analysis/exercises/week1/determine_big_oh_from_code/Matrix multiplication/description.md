For the code snippet below, express the time complexity in terms of a polynomial representation.
Simplify your polynomial representation to derive the time complexity in Big Oh notation.

Note: you can assume the matrices are square.

```python
def multiply_matrix(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
   res = []
   for a_row in range(len(a)):
       res_row=[]
       for b_col in range(len(b[0])):
           res_elem = 0
           for a_col in range(len(a[0])):
               res_elem += a[a_row][a_col] * b[a_col][b_col]
           res_row.append(res_elem)
       res.append(res_row)
   return res
```
