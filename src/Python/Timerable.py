class TimerWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        # If the first argument is an instance of a class, it's likely 'self'
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {self.func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result

def timerable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__class__'):
            return TimerWrapper(func)(*args, **kwargs)
        else:
            return TimerWrapper(func)(*args, **kwargs)
    return wrapper