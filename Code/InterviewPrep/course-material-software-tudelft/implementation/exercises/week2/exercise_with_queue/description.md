Congratulations on your new job as the new warehouse manager at Python Post International!
It is your task to make the warehouse run as efficiently as possible.
Throughout the day trucks arrive to bring packages, which are then routed to one of three terminals based on their label.
Besides regular packages there are also priority packages, which need to be delivered as soon as possible.

Implement the `store` and `collect` functions in the `Warehouse` class so that your warehouse meets the following constraints:
- Your warehouse has 3 terminals.
- Regular packages must be delivered with a minimal delay per package.
- Priority packages must be delivered before regular packages.
    - Their relative order is irrelevant.
- At most 1 data structure is used per terminal.
- All operations are done as efficiently as possible.

<details>
    <summary>Hint: What data structure to use?</summary>
    To store the packages in the warehouse you will need some data structure.
    You should choose between a standard (array-based) list or a linked list.
    Keep in mind what operations you are going to execute on the data structure and their respective complexities,
    as you must provide the most efficient implementation.
    If you want to use a linked list, you can use a
    <a href="https://docs.python.org/3/library/collections.html#collections.deque">deque</a> as a DLL instead of creating one yourself.
</details>
