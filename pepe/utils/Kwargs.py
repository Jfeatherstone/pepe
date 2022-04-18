"""
Decorator functions to simplify how kwargs are passed to pipline functions.
"""
from functools import wraps

def explicitKwargs():
    """
    Decorator that creates an attribute for the decorated function (`explicit_kwargs`)
    which is a list of the keyword arguments that are explicitly passed (and not just
    left to the default value).

    Adapted from:
    https://stackoverflow.com/questions/1408818/getting-the-keyword-arguments-actually-passed-to-a-python-method
    """
    def decorator(function):
        # Have to carry over the doc string, otherwise the documentation
        # won't generate properly
        @wraps(function)
        def inner(*args, **kwargs):
            inner.explicit_kwargs = kwargs.keys()
            return function(*args, **kwargs)
        return inner
    return decorator

