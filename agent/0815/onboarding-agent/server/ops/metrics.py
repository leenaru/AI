from contextlib import contextmanager
import time

@contextmanager
def trace(name: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = (time.time() - t0) * 1000
        print(f"[trace] {name} took {dt:.1f}ms")
