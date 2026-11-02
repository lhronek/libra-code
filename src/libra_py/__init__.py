# ***********************************************************
# * Copyright (C) 2016-2026 Alexey V. Akimov
# * This file is distributed under the terms of the
# * GNU General Public License as published by the
# * Free Software Foundation; either version 3 of the
# * License, or (at your option) any later version.
# * http://www.gnu.org/copyleft/gpl.txt
# ***********************************************************/
import sys

# every libra process writes whole lines: a stdout sent to a file is block-buffered, and the forked pool workers of step 2,
# step 3 and the dynamics share the runner's - the 8 KB chunks of several workers land interleaved mid-line in the common
# log, a child process's output overtakes the lines still in the buffer (the C stdout of the C++ prints is line-buffered by
# liblibra_core's import, a stream without reconfigure - a notebook's or a capture - is left alone)
if hasattr(sys.stdout, "reconfigure") and not getattr(sys.stdout, "line_buffering", False):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (ValueError, OSError):      # a detached or closed stream
        pass


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
