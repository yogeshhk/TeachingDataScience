For cooking many people use a timer clock. The old-fashioned mechanic clocks allow you to manually decrease the time you set.
New digital clocks however do not: once you set the time, you can start, stop or reset the timer.
There is no way to decrease the time.

In the `library` you will find a `Clock` class that represents a mechanical timer.
This timer has a function `add_time` that adds the given time to the clock. Note that if we provide a negative argument,
time will be decreased.
Furthermore there is a function `tick` that represents 1 tick of the clock. We can reset the time with the `reset` function.

You should do the following:

* Implement a custom `NegativeTimeError` class that extends the `Exception` class.
* Implement the class `DigitalClock` that extends the `Clock` class.
* Override the `add_time` function so that it raises a `NegativeTimeError` when the `time` parameter is negative.

<details>
    <summary>Hint: custom exception class</summary>
    You can create your own exception class by defining a class that extends <code>Exception</code>. For example: <br/>
    <pre>class MyOwnError(Exception):
    pass</pre>
</details>

