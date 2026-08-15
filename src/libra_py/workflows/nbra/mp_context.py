"""
.. module:: mp_context
   :platform: Unix, Windows
   :synopsis: Shared multiprocessing context for the nbra workflow modules

Python >= 3.14 defaults to the "forkserver" start method on Linux, which
re-imports the user's driver script in every worker - a typical unguarded
nac.py-style script then re-executes the whole workflow per worker and the
Pool dies with a bootstrapping RuntimeError. Fork keeps the pre-3.14
semantics (workers inherit the parent state). Fork also inherits the
parent's thread state, so the nbra pools are created only after the
BLAS caps quiesce the parent (_with_parent_blas_cap) and workers apply
their own caps. Fork applies to Linux only: it is unsafe on macOS
(the ObjC runtime) and absent on Windows - both use their platform
default.
"""

import multiprocessing as mp
import sys

if sys.platform.startswith("linux"):
    MP_CTX = mp.get_context("fork")
else:
    MP_CTX = mp
