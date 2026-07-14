"""
.. module:: blas_threads
   :platform: Unix, Windows
   :synopsis: Best-effort control of the thread pool of the BLAS library
       loaded in this process (OpenBLAS via threadpoolctl or directly)

The cap governs the library's single thread pool, which serves both its
BLAS and its bundled LAPACK routines (zheevd etc.).

The BLAS behind numpy and liblibra threads on its own, independently of any
process- or OpenMP-level parallelism in libra. On the small per-step matrices
handled by the NBRA workflows and the TSH dynamics the thread-pool handoff
rivals the arithmetic itself, so capping the pool (usually to 1 thread)
inside the compute region - and restoring the previous value afterwards, so
the rest of the process keeps its threading - is the right default.
"""

import contextlib


@contextlib.contextmanager
def blas_thread_limit(n):
    """Best-effort cap on the BLAS/OpenMP thread count inside the with-block.

    Uses threadpoolctl when installed, otherwise talks to the loaded OpenBLAS
    directly; restores the old value on exit. A falsy limit (None or 0)
    leaves the pool untouched.

    Args:
        n ( int ): the thread cap; None or 0 = do nothing
    """
    if not n:
        yield
        return
    n = max(1, int(n))
    limiter = None
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=n)
    except ImportError:
        pass
    if limiter is not None:
        with limiter:
            yield
        return
    saved = []
    try:
        import ctypes
        with open('/proc/self/maps') as f:
            libs = sorted({line.split()[-1] for line in f
                           if 'openblas' in line.lower() and '.so' in line.split()[-1]})
        for path in libs:
            try:
                handle = ctypes.CDLL(path)     # already loaded - same handle
                get = getattr(handle, 'openblas_get_num_threads', None)
                set_ = getattr(handle, 'openblas_set_num_threads', None)
                if set_ is not None:
                    saved.append((set_, get() if get is not None else 0))
                    set_(n)
            except OSError:
                pass
    except OSError:
        pass
    try:
        yield
    finally:
        for set_, old in saved:
            if old > 0:
                set_(old)
