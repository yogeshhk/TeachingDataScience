####1) Corresponding run time equation for method_y and explanation of all terms

Here we have to realize that the worst case is achieved when the element of interest is not in the queue and we have to go through the whole structure to find that out. Thus we have the following equation:

\\(T(n)=c_0 + nc_1 + nc_2\\)

Which is equal to:

\\(T(n)=c_0 + nc_3\\)
\\(where \quad c_3 = c_1 + c_2\\)

Where:

- \\(n\\) is the size of the queue `q`
- \\(c_0\\) represents the constant-time operations on lines 3, 4, 10, 12 (declare temporary queue, initialize the first and second loop, return statement)
- \\(c_1\\) represents the constant-time operations on lines 5, 6, 9 (dequeue the first value from Queue `q` and store it in current, check if that's the value we are looking for and if not enqueue it in Queue `temp`)
- \\(c_2\\) represents the constant-time operations on lines 11 (dequeue the first value from Queue `temp` and enqueue it in Queue `q`)

####2) State the run time complexity in terms of Big Oh notation.

After simplification, the constant terms and factors can be disregarded as n grows to infinity. This leaves us with \\(T(n) = n\\).
This is linear growth. Thus, the run time complexity is \\(\mathcal{O}(n)\\).

