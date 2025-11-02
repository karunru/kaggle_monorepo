import functools
import logging
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str, log: bool = True):
    t0 = time.time()
    msg = f"[{name}] start"
    print(msg)
    # slack_notify(msg)

    if log:
        logging.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    print(msg)
    # slack_notify(msg)

    if log:
        logging.info(msg)


def timer_decorator(name: str = None, log: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            t0 = time.time()
            msg = f"[{timer_name}] start"
            print(msg)

            if log:
                logging.info(msg)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                msg = f"[{timer_name}] done in {time.time() - t0:.2f} s"
                print(msg)

                if log:
                    logging.info(msg)

        return wrapper

    return decorator
