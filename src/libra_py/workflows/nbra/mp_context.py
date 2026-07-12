"""
.. module:: mp_context
   :platform: Unix, Windows
   :synopsis: Shared multiprocessing context for the nbra workflow modules

Python >= 3.14 defaults to the "forkserver" start method on Linux, which
re-imports the user's driver script in every worker - a typical unguarded
nac.py-style script then re-executes the whole workflow per worker and the
Pool dies with a bootstrapping RuntimeError. Fork keeps the pre-3.14
semantics (workers inherit the parent state); where "fork" is unavailable
(e.g. Windows), fall back to the default context.
"""

import multiprocessing as mp

try:
    MP_CTX = mp.get_context("fork")
except ValueError:
    MP_CTX = mp
