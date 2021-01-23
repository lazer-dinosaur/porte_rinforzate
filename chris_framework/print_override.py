import builtins as __builtin__
import numpy as np
import time

def pp(a,name='',enumer=False):
    a = np.round(a,2)
    s = ''
    for i,aa in enumerate(a):
        if enumer:
            s += f'{i:>3}: {aa:>5.2f}, '
        else:
            s += f'{aa:>5.2f}, '
        if not (i+1)%6:
            s += '\n'
    __builtin__.print(s,name)

def print(*args, **kwargs):
    array = kwargs.pop('a',False)
    enumer = kwargs.pop('enumer',ENUMER)
    name = kwargs.pop('n','')
    timer = kwargs.pop('timer',TIMER)
    override = kwargs.pop('o',False)
    if override:
        if array:
            if timer:
                time.sleep(0.001)
            return pp(args[0],name,enumer)
        return __builtin__.print(*args, **kwargs)

    try:
        PRINT
    except NameError:
        if array:
            if timer:
                time.sleep(0.001)
            return pp(args[0],name,enumer)
        return __builtin__.print(*args, **kwargs)

    if PRINT:
        if timer:
            time.sleep(0.001)
        if array:
            return pp(args[0],name,enumer)
        return __builtin__.print(*args, **kwargs)

PRINT = 1
ENUMER = True
TIMER = 1