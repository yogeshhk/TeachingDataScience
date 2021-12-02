An example solution may look like this:
We can make use of a priority queue. That way the normal customers will wait similar amount of time, proportional to their
arrival. Since we need to make sure that the regular customers wait as little as possible, we can give them higher priority
and every time they come to the store they will be moved to the front of the queue. In a general case a queue will suffice 
but here we want to treat regular customers differently, thus we need the priority queue.