def simple_decorator(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")

def decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print(f"Function called with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function returned: {result}")
        return result
    return wrapper

@decorator_with_args
def add(a, b):
    return a + b

def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    import time
    time.sleep(0.1)
    return "Done"

def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print("=== Simple Decorator ===")
    say_hello()
    
    print("\n=== Decorator with Arguments ===")
    result = add(5, 3)
    
    print("\n=== Timer Decorator ===")
    slow_function()
    
    print("\n=== Decorator Factory ===")
    greetings = greet("Alice")
    for greeting in greetings:
        print(greeting)