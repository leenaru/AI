from contextlib import contextmanager
import time
@contextmanager
def trace(name:str):
 t=time.time(); yield; print(f"[trace] {name} took {(time.time()-t)*1000:.1f}ms")
