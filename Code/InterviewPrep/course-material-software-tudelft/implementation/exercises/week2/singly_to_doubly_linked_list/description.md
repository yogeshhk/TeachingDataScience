The local grocery store has launched a new app, in hope of winning back some of the customers from their big competitors. This new app not only allows you to buy all the ingredients for a meal with one click, but it also includes a step by step guide of how to cook this meal.

However the developer of this app forgot to do his user research beforehand, he implemented the app without knowing that the majority of the users will be students, who might or might not be slightly impatient and forgetful.

Now what happened is that he used a singly linked list to implement the guide, where every step in the cooking process is one node in the linked list. Remember that a singly linked list only allows you to traverse the list in one direction, which means that the user will not be able to go back to the previous step, without starting back from the beginning.

Your job is to fix this, by changing the existing singly linked list to a doubly linked list, which allows for traversing backwards.

The developer has left some hints for you.

<details>
    <summary>Nodes</summary>
       In a singly linked list, a node has a reference to the next node. However for doubly linked list, we also need to be able to traverse backwards. For this purpose a node should not only have a reference to its next node, but also a reference to its previous node.
       The developer has already included this variable for you, make sure to use it.
</details>

<details>
    <summary>Initializing the linked list</summary>
        For the singly linked list we only need to initialize a pointer to head, but perhaps we could include a pointer to the tail as well.
        The developer has already included this variable for you, make sure to use it.
</details>

<details>
    <summary>Add and remove functions</summary>
        The functionality for adding and removing is already there for a singly linked list, what else do we need to do for a doubly linked list?
</details>