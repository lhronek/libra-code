# ***********************************************************
# * Copyright (C) 2016-2026 Alexey V. Akimov
# * This file is distributed under the terms of the
# * GNU General Public License as published by the
# * Free Software Foundation; either version 3 of the
# * License, or (at your option) any later version.
# * http://www.gnu.org/copyleft/gpl.txt
# ***********************************************************/
import os
import sys
import atexit
import threading


class _WholeLines:
    """sys.stdout and sys.stderr of every libra process: a line goes out whole, in one write, when its newline arrives. print writes a
    line in pieces (the text, the end), the forked pool workers of step 2, step 3 and the dynamics share the runner's
    stdout, and `python -u` makes it unbuffered - the pieces of several workers landed interleaved mid-line in the common
    log, and two threads of one process merged their lines. Each thread's pieces wait in its own buffer until the newline,
    then the whole line is written and flushed under one lock, whatever the stream's own mode (unbuffered, a file's block
    buffer, a terminal) - everything else is the wrapped stream's. The C stdout and stderr of the C++ prints are
    line-buffered by liblibra_core's import."""

    _libra_whole_lines = True

    def __init__(self, stream):
        self._stream = stream
        self._parts = {}
        self._lock = threading.Lock()

    def write(self, text):
        tid = threading.get_ident()
        buf = self._parts.pop(tid, "") + text
        cut = buf.rfind("\n")
        if cut >= 0:
            with self._lock:
                self._stream.write(buf[:cut + 1])
                self._stream.flush()
            buf = buf[cut + 1:]
        if buf:
            self._parts[tid] = buf
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        buf = self._parts.pop(threading.get_ident(), "")
        with self._lock:
            if buf:
                self._stream.write(buf)
            self._stream.flush()

    def flush_all(self):
        with self._lock:
            for buf in self._parts.values():
                self._stream.write(buf)
            self._parts.clear()
            self._stream.flush()

    def _after_fork(self):
        self._lock = threading.Lock()      # the parent's threads, one of them maybe holding the lock, do not exist here
        self._parts = {}

    def __getattr__(self, name):
        return getattr(self._stream, name)


for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name)
    if _stream is not None and not getattr(_stream, "_libra_whole_lines", False):
        _wrapped = _WholeLines(_stream)
        setattr(sys, _name, _wrapped)
        atexit.register(_wrapped.flush_all)
        os.register_at_fork(after_in_child=_wrapped._after_fork)
del _name, _stream


def __getattr__(name):
    # lazy: importing libra_py must not load the C++ extension (banner
    # side effect, segfault surface under a mismatched python)
    if name == "__version__":
        try:
            from liblibra_core import __version__ as v
        except Exception:
            v = "unknown"
        return v
    raise AttributeError(F"module 'libra_py' has no attribute '{name}'")

__all__ = ["models",
           "workflows",
           "recipes",
           "acf",
           "autoconnect",
           "build",
           "citools",
           "cube_file_methods",
           "data_conv",
           "data_io",
           "data_outs",
           "data_read",
           "data_savers",
           "data_stat",
           "data_visualize",
           "dyn",
           "dynamics",
           "dynamics_plotting",
           "eigensolvers",
           "fgr_py",
           "fit",
           "fix_motion",
           "ft",
           "gaussian_kernel_algorithm",
           "grids",
           "hpc_utils",
           "hungarian",
           "init_ensembles",
           "init_system",
           "initial_conditions",
           "influence_spectrum",
           "libra_executable",
           "md_align",
           "LoadGAFF",
           "LoadMMFF94",
           "LoadMolecule",
           "LoadPT",
           "LoadTRIPOS",
           "LoadUFF",
           "molden_methods",
           "namd",
           "normal_modes",
           "nve_md",
           "orthogonalizations",
           "packages",
           "parse_gamess",
           "pdos",
           "probabilities",
           "regexlib",
           "scan",
           "scf",
           "spectrum",
           "trpes",
           "tsh_stat",
           "unavoided",
           "units",
           "vesta2qe"
           ]
