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
    if not allowed:                                   # an entry reading no key has nothing to suggest
        return None
    best = min(allowed, key=lambda x: (edit_distance(key.lower(), x.lower()), x))
    return best if edit_distance(key.lower(), best.lower()) <= within else None


def unknown_keys(params, allowed, skip=()):
    """The keys of `params` outside `allowed` and `skip`, sorted, and the hints "key -> nearest?" for those within two edits"""
    unknown = sorted(k for k in params if k not in allowed and k not in skip)
    hints = []
    for k in unknown:
        best = nearest_key(k, allowed)
        if best is not None:
            hints.append(F"{k} -> {best}?")
    return unknown, hints


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
    unknown, hints = unknown_keys(params, allowed, skip)
    if not unknown:
        return
    raise ValueError(F"{where}: {carrier} carries keys that {nobody_reads}: {unknown}"
                     + (F" (did you mean {hints}?)" if hints else "")
                     + " - an unknown key is ignored silently, so a misspelled key must not pass")


def warn_unknown_keys(params, allowed, where, carrier, nobody_reads, skip=()):
    """Print a WARNING naming the keys of `params` outside `allowed` (and `skip`) with their nearest known keys, and return
    them - for the entries whose callers share one dict with other entries (the plot dicts of the tutorials), where a
    refusal would reject the other entries' keys

    Args: as refuse_unknown_keys
    """
    unknown, hints = unknown_keys(params, allowed, skip)
    if unknown:
        print(F"WARNING in {where}(): {carrier} carries keys that {nobody_reads}: {unknown}"
              + (F" (did you mean {hints}?)" if hints else ""))
    return unknown


def restrict(params, allowed):
    """Return a new dict holding the items of `params` whose keys are in `allowed` - the copy an in-library caller hands an
    entry that refuses unknown keys when the caller's own dict carries more; the entry's defaults land in the copy, not in
    `params`

    Args:
        params ( dict ): the caller's dict
        allowed ( iterable of str ): the entry's key vocabulary

    Returns:
        dict: the restricted copy
    """
    return {k: v for k, v in params.items() if k in allowed}
