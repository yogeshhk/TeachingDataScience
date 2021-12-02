A common activity for students during a lecture is playing BINGO.
However, they don't play the regular boring game with the numbered balls.
Instead, their BINGO sheets contain in every square an event that might
(or might not) happen during a lecture, and once the event happens, this
square can be crossed off.

Since you'd also like to pay attention during the lecture while playing
this game, you decide to build a program that automatically plays BINGO for
you.
Given a BINGO sheet with a grid of \\(n \times n\\) squares, and a list of
events that happen during the lecture, how many events will pass before you
can shout "BINGO!"?

Note that you may shout "BINGO!" whenever you crossed off all squares in
any row, column, or diagonal of the grid.
The middle square in your grid can always be crossed off for free.

#### Input
One line containing two integers: one odd integer \\(n\\), with
\\(1 \leq n \leq 10^3\\), and one integer \\(m\\), with \\(0 \leq m \leq 10^6\\).
\\(n\\) lines, each containing \\(n\\) space-separated events.
Event names consist of only alphanumeric characters, and every event only
happens at most once.
\\(m\\) lines, with on every line an event that happens during the lecture.

#### Output
One integer, indicating after how many events you can shout "BINGO!".
If you cannot shout "BINGO!" during this lecture, output a sad smiley
face: `:-(`

#### Examples

For each example, the first block is the input and the second block is the corresponding output.
##### Example 1
```
3 10
WordMispronounced LoudMic Gaming
Sleeping FreeLunch Latecomer
2MinutesSilence TeacherAngry Eating
MicBreaks
Sleeping
SlideshowContainsMistake
WordMispronounced
Gaming
PhoneRings
Eating
Latecomer
TeacherAngry
2MinutesSilence
```
```
7
```
##### Example 2
```
3 5
EventA EventB EventC
EventD EventE EventF
EventG EventH EventI
EventJ
EventA
EventK
EventB
EventD
```
```
:-(
```

Source: FPC 2018
