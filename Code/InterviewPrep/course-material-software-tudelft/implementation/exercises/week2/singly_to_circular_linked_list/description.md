Hands are sweating, gears (in your head) are turning, it's the last day before the final exam of Algorithms and Data structures. You are studying in the library of TU Delft, while listening to music. Your phone is set on repeat with your favorite playlist. 

After merely one hour, you noticed that the music has stopped, your playlist is not repeating itself! One of the recent updates for your music player, has (un)intentionally broken the auto repeat functionality. 

Normally you would rant for a few seconds, before putting your favorite playlist on again and continue to study. However, this time you have decided to take matters into your own hands, you will fix this bug in your music player.

In one of the recent updates, the developers have changed the playlist to use a singly linked list, remember that the last node in the singly linked list points towards nothing (`None`). Your task is to change the singly linked list to a circular linked list, this way we can go back to the head from the last node.

<details>
    <summary>Hint</summary>
        Note that the singly linked list, already has a pointer towards the <code>tail</code>. Your circular linked list should also keep track of the <code>head</code> and <code>tail</code>.
        
        Try to figure out for each method, what should be changed to achieve the behaviour of a circular linked list. Not all methods will behave differently.
</details>