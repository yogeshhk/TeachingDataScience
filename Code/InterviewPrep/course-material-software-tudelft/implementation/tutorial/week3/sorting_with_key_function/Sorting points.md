In this exercise you have to implement two functions on sorting 2D points. The `Point` class is given in the library.

A list of numeric `xs` can be sorted using `sorted(xs)`, but for a custom class like `Point` we need to define what to sort on.
This is done by adding a specification for the key to sort on: `sorted(xs, key=lamdba point: point.x)`.
The lambda function returns the `x` attribute for a point.
This is done for all points in `xs` and sorting happens based on this output.

Your task is to implement these two functions:

- `order_by_x` returns a sorted list of the points, based on their x coordinate only. 
If two points have the same x coordinate, their respective order is irrelevant.

- `order_by_x_then_y` returns a sorted list of the points, based on their x coordinate.
If two points have the same coordinate, their respective order is based on the y coordinate.

<details>
    <summary>Sorting on multiple attributes</summary>
    Python's operator package has a built in function called attrgetter. We can use it as follows:
    <pre>sorted(xs, key=attrgetter("x"))</pre>
    This does the same as the example with a lambda given earlier.
    The advantage of the attrgetter is that you can also call it with multiple attributes, which are evaluated in that order:
    <pre>sorted(xs, key=attrgetter("x", "y"))</pre> sorts on "x" and uses "y" when it encounters the same value for "x" on two elements.
    <br/><br/>
    Try to implement the second function with the attrgetter and after you've done that
    try if you can make it work by reusing the first function as an extra challenge.
</details>
