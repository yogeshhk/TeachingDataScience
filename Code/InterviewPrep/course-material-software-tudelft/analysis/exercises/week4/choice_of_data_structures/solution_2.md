An example solution may look like this:
An appropriate data structure would be a stack. Going through the string we could `push` whenever we encounter an opening 
bracket and `pop` when it's a closing one. According to the item popped we continue if it matches the bracket we were at 
in the string or conclude that the string is not a valid balanced match otherwise. The use of the stack makes the search 
and comparison the most efficient for this problem as it allows us to store the previously encountered opening brackets
by the LIFO principle and compare the current closing bracket to the last opening one, thus verify that the brackets
are correctly nested within each other.