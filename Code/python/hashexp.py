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


def timeit_timeout(command, setup='', TIMEOUT=10, default=None, err=.05):
    return timeout(Timeit, args=command, kwargs={'setup': setup},
                   TIMEOUT=TIMEOUT, default=default, err=err)


# END OF CODE SNIPPET

class Bad:

    def __init__(self, x):
        self.x = x

    def __hash__(self):
        return 1


def badhash():
    d = dict()
    for i in range(1000):
        d[Bad(i)] = i


class Good:

    def __init__(self, x):
        self.x = x

    def __hash__(self):
        return hash(self.x)


def goodhash():
    d = dict()
    for i in range(1000):
        d[Good(i)] = i


print("bad hash", timeit_timeout(badhash))  # 1000x times = 7.6 s
print("good hash", timeit_timeout(goodhash))  # 1000x times = 0.4 s
