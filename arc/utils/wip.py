import unittest
from functools import wraps

def work_in_progress(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            raise unittest.SkipTest("WIP test failed: " + str(e))
        raise AssertionError("test passed but marked as work in progress")
    return wrapper
