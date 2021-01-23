import time
import inspect
import datetime
from functools import wraps
#####################################################
import line_profiler
import atexit
timing_p = line_profiler.LineProfiler()
atexit.register(timing_p.print_stats, stripzeros=True)
#####################################################

def timing(reps=1, text='', longrunning=False, **keywords):
    '''
    Use:
    @timing.timing()
    or:
    @timing.timing_p
    for line profiler

    Additionally:
    Set keyword, value pairs to over-ride any parameter in the decorated function, ie:

    @timing(10, cat=77, draw=True)
    def hat(cat, draw=False):
        print(cat, draw)

    hat(99,False)
    '''
    def real_decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            args = list(args)
            for i, a in enumerate(inspect.getfullargspec(f)[0][:len(args)]):
                if a in keywords:
                    args[i] = keywords.pop(a)
            kw.update(keywords)

            start = time.time()
            for i in range(reps):
                result = f(*args, **kw)
            end = time.time()
            time_taken = (end - start) / reps
            if text:
                print(text)
            print(f'{f.__name__:30}: {time_taken:10.6f}s, {1 / time_taken:5.1f}fps, {str(datetime.timedelta(seconds=time_taken))}')
            return result
        return wrapper
    return real_decorator

