# ***********************************************************
# * Copyright (C) 2026 Lukas Hronek
# * This file is distributed under the terms of the
# * GNU General Public License as published by the
# * Free Software Foundation; either version 3 of the
# * License, or (at your option) any later version.
# * http://www.gnu.org/copyleft/gpl.txt
# ***********************************************************/
"""
.. module:: params_check
   :platform: Unix, Windows
   :synopsis: the keys of a parameter dictionary against the keys its consumers read - an unknown key is refused naming
       it and the nearest known key, because the consumers ignore an unknown key silently (comn.check_input fills the
       defaults and keeps every extra key, the engine's set_parameters walks its key chain without an else)

.. moduleauthor:: Lukas Hronek
"""


def edit_distance(a, b):
    """The Levenshtein distance of two strings"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest_key(key, allowed, within=2):
    """The allowed key nearest to `key` (case-insensitive edit distance) when it lies within `within` edits, else None"""
    best = min(allowed, key=lambda x: (edit_distance(key.lower(), x.lower()), x))
    return best if edit_distance(key.lower(), best.lower()) <= within else None


def refuse_unknown_keys(params, allowed, where, carrier, nobody_reads, skip=()):
    """Raise ValueError when `params` carries a key outside `allowed` (and outside `skip`), naming the keys and their
    nearest known keys

    Args:
        params ( dict ): the dictionary given by the caller
        allowed ( set ): the keys the consumers read (the defaults, the critical keys, the engine's parameters ...)
        where ( str ): the function refusing, for the message - "run_dynamics", "heom.run_dynamics"
        carrier ( str ): the dictionary's name in the message - "dyn_params"
        nobody_reads ( str ): the clause naming who does not read the unknown keys - "neither the engine nor
            run_dynamics reads", "run_dynamics does not read"
        skip ( iterable ): keys neither allowed nor refused here (a caller handling them on its own)
    """
    unknown = sorted(k for k in params if k not in allowed and k not in skip)
    if not unknown:
        return
    hints = []
    for k in unknown:
        best = nearest_key(k, allowed)
        if best is not None:
            hints.append(F"{k} -> {best}?")
    raise ValueError(F"{where}: {carrier} carries keys that {nobody_reads}: {unknown}"
                     + (F" (did you mean {hints}?)" if hints else "")
                     + " - an unknown key is ignored silently, so a misspelled key must not pass")
