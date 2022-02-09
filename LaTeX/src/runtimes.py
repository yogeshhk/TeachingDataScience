#!/usr/bin/python

# CODE SNIPPET FROM: https://stackoverflow.com/a/20862593 by cdhagmann


import multiprocessing as mp
import timeit


def timeout(func, args=(), kwargs=None, TIMEOUT=10, default=None, err=.05):
    if hasattr(args, "__iter__") and not isinstance(args, basestring):
        args = args
    else:
        args = [args]
    kwargs = {} if kwargs is None else kwargs

    pool = mp.Pool(processes=1)

    try:
        result = pool.apply_async(func, args=args, kwds=kwargs)
        val = result.get(timeout=TIMEOUT * (1 + err))
    except mp.TimeoutError:
        pool.terminate()
        return default
    else:
        pool.close()
        pool.join()
        return val


def Timeit(command, setup=''):
    return timeit.Timer(command, setup=setup).timeit(1000)


def timeit_timeout(command, setup='', TIMEOUT=3, default=None, err=.05):
    return timeout(Timeit, args=command, kwargs={'setup': setup},
                   TIMEOUT=TIMEOUT, default=default, err=err)


# END OF CODE SNIPPET


n = 1


def constant():
    a = 10 + 10


def linear():
    s = 0
    for i in range(n):
        s += 1
    return s


def quadratic():
    s = 0
    for i in range(n):
        for j in range(n):
            s += 1
    return s


def cubic():
    s = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s += 1
    return s


def exp():
    s = 0
    m = 1
    for i in range(n):
        m *= 2
        for j in range(m):
            s += 1
    return s


def factorial():
    s = 0
    m = 1
    for i in range(n):
        m *= n
        for j in range(m):
            s += 1
    return s


def main():
    global n
    for i in range(1, 10):
        n = i
        print("constant", n, timeit_timeout(constant))
        print("linear", n, timeit_timeout(linear))
        print("quadratic", n, timeit_timeout(quadratic))
        print("cubic", n, timeit_timeout(cubic))
        print("exp", n, timeit_timeout(exp))
        print("factorial", n, timeit_timeout(factorial))
    for i in range(2, 6):
        n = 10 ** i
        print("constant", n, timeit_timeout(constant))
        print("linear", n, timeit_timeout(linear))
        print("quadratic", n, timeit_timeout(quadratic))
        print("cubic", n, timeit_timeout(cubic))
        # print("exp", n, timeit_timeout(exp))
        # print("factorial", n, timeit_timeout(factorial))


main()
