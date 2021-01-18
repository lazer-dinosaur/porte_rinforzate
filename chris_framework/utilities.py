import time

cache_store = {}

def cache(key_ix=''):
    def real_decorator(function):
        def wrapper(*args,**kw):
            key = f'{key_ix}:{str(args[1])}'
            result = cache_store.get(key)
            if result:
                return result
            result = function(*args,**kw)
            cache_store[key] = result
            return result
        return wrapper
    return real_decorator
