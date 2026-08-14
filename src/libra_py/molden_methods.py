# *********************************************************************************
# * Copyright (C) 2021 Mohammad Shakiba, Alexey V. Akimov
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: molden_methods
   :platform: Unix, Windows
   :synopsis: This module implements functions for processing the molden files printed out for molecular orbitals (works properly for CP2K outputs).
.. moduleauthors::
       Mohammad Shakiba, Alexey V. Akimov

"""

import os
import sys
import numpy as np
import time

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

from libra_py import data_outs, data_read
from libra_py import units
import math


# Molden cartesian component orders (the molden format specification, l <= 4)
# mapped to the descending-lexicographic canonical order the libint wrappers
# emit: entry k = the molden position feeding canonical slot k
_MOLDEN_CART_TO_LEX = {
    0: [0],
    1: [0, 1, 2],                                              # x, y, z
    2: [0, 3, 4, 1, 5, 2],
    3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2],
    4: [0, 3, 4, 9, 12, 10, 5, 13, 14, 7, 1, 6, 11, 8, 2],
}


def molden_convention(lines, l_vals):
    """Derive the per-l purity map of a molden file and cross-check the basis
    size. Molden defaults to CARTESIAN d/f/g; the [5D]/[5D7F]/[5D10F]/[7F]/
    [9G] markers switch per l, and a markerless cartesian file carries no
    marker at all - the [MO] coefficient indices are the ground truth.

    Args:
        lines (list): The molden file lines.
        l_vals (list): Angular momenta of the shells, in file order.

    Returns:
        tuple: (purity, expected, alternate, file_nbasis) - the {l: bool
        spherical} map, the basis size implied by the markers, the size under
        the flipped d/f/g convention, and the largest AO index seen in the
        [MO] section (None if the file has no [MO] coefficients).
    """
    d_sph = f_sph = g_sph = False
    for ln in lines:
        u = ln.upper().replace(' ', '')
        if u.startswith('[5D7F]') or u.startswith('[5D]'):
            d_sph = True
            f_sph = True                       # molden: [5D] implies 7F
        if u.startswith('[5D10F]'):
            d_sph = True
            f_sph = False
        if u.startswith('[7F]'):
            f_sph = True
        if u.startswith('[6D10F]') or u.startswith('[6D]'):
            d_sph = False
        if u.startswith('[10F]'):
            f_sph = False
        if u.startswith('[9G]'):
            g_sph = True
    # p follows the d convention: spherical files keep the historical pure-p
    # route, cartesian files use cartesian p (the same x, y, z order)
    purity = {0: True, 1: d_sph, 2: d_sph, 3: f_sph, 4: g_sph}

    def _size(l, sph):
        return 2 * l + 1 if (sph or l < 2) else (l + 1) * (l + 2) // 2

    expected = sum(_size(l, purity[l]) for l in l_vals)
    alternate = sum(_size(l, not purity[l] if l >= 2 else purity[l])
                    for l in l_vals)
    file_n = 0
    in_mo = False
    for ln in lines:
        if '[MO]' in ln.upper():
            in_mo = True
            continue
        if in_mo:
            t = ln.split()
            if len(t) == 2:
                try:
                    file_n = max(file_n, int(t[0]))
                except ValueError:
                    pass
    return purity, expected, alternate, (file_n if file_n > 0 else None)


def _cart_coeff_scale(l_vals, purity):
    """Per-AO coefficient scale, in the lexicographic canonical order: molden
    cartesian coefficients refer to shared-shell-scaled components (the pure
    power carries the shell normalization), so a component x^i y^j z^k needs
    sqrt((2l-1)!!/((2i-1)!!(2j-1)!!(2k-1)!!)) to refer to the unit-normalized
    cartesians the libint wrappers emit (sqrt(3) for xy-type, sqrt(15) for
    xyz; empirically exact on CP2K GTO_KIND CARTESIAN output)."""
    def dfact(n):
        r = 1.0
        while n > 1:
            r *= n
            n -= 2
        return r

    scale = []
    for l in l_vals:
        if purity.get(l, True) or l < 2:
            scale += [1.0] * (2 * l + 1 if l != 1 else 3)
        else:
            for i in range(l, -1, -1):
                for j in range(l - i, -1, -1):
                    k = l - i - j
                    cca = dfact(2 * i - 1) * dfact(2 * j - 1) * \
                        dfact(2 * k - 1) / dfact(2 * l - 1)
                    scale.append(1.0 / np.sqrt(cca))
    return np.array(scale)


def molden_purity(molden_filename, l_vals):
    """The per-l spherical-purity map of a molden file, with the coefficient
    count as the ground truth for markerless cartesian files."""
    with open(molden_filename) as f:
        lines = f.readlines()
    purity, expected, alternate, file_n = molden_convention(lines, l_vals)
    if file_n is not None and file_n == alternate and file_n != expected:
        purity = {k: (not v if k >= 2 else v) for k, v in purity.items()}
        purity[1] = purity[2]
    return purity


def _molden_gto_l_vals(lines):
    """Angular momenta of the [GTO] shells, in file order (pre-pass so the
    convention can be validated before the shells are built)."""
    letters = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    start = end = None
    for i, ln in enumerate(lines):
        u = ln.upper().replace(' ', '')
        if u.startswith('[GTO]'):
            start = i
        elif start is not None and u.startswith('[') and i > start:
            end = i
            break
    if end is None:
        end = len(lines)
    l_vals = []
    for i in range(start + 1, end):
        t = lines[i].split()
        if len(t) >= 2 and t[0].lower() in letters:
            l_vals.append(letters[t[0].lower()])
    return l_vals


def molden_file_to_libint_shell(molden_filename: str, is_spherical: bool, is_periodic=False,
                                cell=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), R_vec=np.array([0, 0, 0])):
    """
    This function gets the molden file and returns the shell for use with
    libint to compute the atomic orbital overlaps.

    Args:

        molden_filename (string): The name of the molden file.

        is_spherical (bool): The Gaussian cartesian or spherical orbital flag.

    Returns:

        shell (shell data type from libint): The shell containing the exponents and contraction
                                             coefficients for each atom, their coordinates in Bohr,
                                             and their angular momentum values.
        l_vals_all (list): The angular momentum values for all the atoms and basis set in order.
                           This is used later to resort the eigenvectors in molden file.

    """
    # Reading the lines of the molden file
    file = open(molden_filename, 'r')
    lines = file.readlines()
    file.close()

    # Derive the file's convention (markers + the [MO] coefficient-count
    # cross-check: molden defaults to cartesian and a cartesian CP2K molden
    # is MARKERLESS, so counts are the ground truth) and validate the
    # caller's flag against it
    _l_pre = _molden_gto_l_vals(lines)
    purity, _expected, _alternate, _file_n = molden_convention(lines, _l_pre)
    if _file_n is not None and _file_n != _expected:
        if _file_n == _alternate:
            purity = {k: (not v if k >= 2 else v) for k, v in purity.items()}
            purity[1] = purity[2]
        else:
            raise ValueError(
                F"molden_file_to_libint_shell: {molden_filename} has "
                F"{_file_n} AO coefficient rows per MO, but the section "
                F"markers imply {_expected} (flipped d/f/g convention: "
                F"{_alternate}) - inconsistent molden file")
    _lge2 = [l for l in set(_l_pre) if l >= 2]
    _all_sph = all(purity[l] for l in _lge2) if _lge2 else True
    if is_spherical and not _all_sph:
        raise ValueError(
            F"molden_file_to_libint_shell: {molden_filename} carries "
            "CARTESIAN d/f/g shells (markers/coefficient count) but "
            "is_spherical=True was requested - pass is_spherical=False")
    if (not is_spherical) and _all_sph and _lge2:
        raise ValueError(
            F"molden_file_to_libint_shell: {molden_filename} is a spherical "
            "([5D7F]-class) file but is_spherical=False was requested - "
            "pass is_spherical=True")

    # The characters in the molden file start with '[' including
    # coordinates, basis sets, eigenvectors, etc. We store them
    # into chars to move between them in a for loop
    chars = []
    for i in range(len(lines)):
        if lines[i].replace(' ', '')[0] == '[':
            chars.append(i)

    # The coordinates of the structure, these are in Bohr value
    coords = []
    for i in range(chars[1] + 1, chars[2]):
        # Each line in the coordinates part
        tmp = lines[i].split()
        # not necessary but let's keep it, might be useful in future
        atomic_number = int(tmp[2])
        # The x, y, and z coordinates
        x = float(tmp[3])
        y = float(tmp[4])
        z = float(tmp[5])
        if is_periodic:
            translation = np.dot(R_vec, cell)
            x += translation[0]
            y += translation[1]
            z += translation[2]
        # append all in the coords
        coords.append([atomic_number, x, y, z])

    """
    The molden format for basis set is the same as below:
     [GTO]
        1       0        * this line
                         s       6    1.00 * three_line
                                                       106.389717      -0.004151
                                                        19.510792      -0.020670
                                                         5.482954      -0.051503
                                                         0.784075       0.334627
                                                         0.355861       0.562106
                                                         0.169708       0.171299
                         p       6    1.00 * three_line
                                                        19.697346       0.007924
                                                         5.136667       0.051441
                                                         1.837949       0.189840
                                                         0.768298       0.404986
                                                         0.351318       0.401236
                                                         0.166091       0.105186

        2       0        * atom_line
                         s       6    1.00 * three_line
                                                       106.389717      -0.004151
                                                        19.510792      -0.020670
                                                         5.482954      -0.051503
                                                         0.784075       0.334627
                                                         0.355861       0.562106
                                                         0.169708       0.171299
                         p       6    1.00 * three_line
                                                        19.697346       0.007924
                                                         5.136667       0.051441
                                                         1.837949       0.189840
                                                         0.768298       0.404986
                                                         0.351318       0.401236
                                                         0.166091       0.105186

    What we need to do is to first find the initial lines I showed with a star* which show
    the basis set for each atom number.
    """
    # atom lines
    atom_line = []
    # first need to set up a counter to check whether
    # it can be an integer or not. We move between [GTO] and [MO]
    # in the file.
    c = 0
    for i in range(chars[2] + 1, chars[3]):
        try:
            # the first value, if it can be turned into
            # an integer value
            first_val = int(lines[i].split()[0])
            # counter starts to increase now.
            c += 1
            # if it was equal to counter add it to the list
            if first_val == c:
                atom_line.append(i)
        except BaseException:
            pass

    # Setting up the counters
    # the counter for counting the atoms
    # necessary to initialize the shell for the first one
    # then for others we add to the initialized shell
    # using add_to_shell
    c = 0
    # coords counter
    h1 = 0
    # all the angular momentum values for all atoms
    l_vals_all = []
    for i in range(len(atom_line)):
        tmp = []
        # Lines with three elements in them as is shown above
        three_line = []
        # the angular momentum values for each atom
        l_vals = []
        if i == len(atom_line) - 1:
            # next atom_line
            end_l = chars[3]
        else:
            # next atom_line
            end_l = atom_line[i + 1]
        # now loop over them and append the angular momentum values
        # for each atom.
        for j in range(atom_line[i], end_l):
            if len(lines[j].split()) == 3:
                three_line.append(j)
                # if s orbital
                if lines[j].split()[0] == 's':
                    l_vals.append(0)
                    l_vals_all.append(0)
                # if p orbital
                elif lines[j].split()[0] == 'p':
                    l_vals.append(1)
                    l_vals_all.append(1)
                # if d orbital
                elif lines[j].split()[0] == 'd':
                    l_vals.append(2)
                    l_vals_all.append(2)
                # if f orbital
                elif lines[j].split()[0] == 'f':
                    l_vals.append(3)
                    l_vals_all.append(3)
                # if g orbital
                elif lines[j].split()[0] == 'g':
                    l_vals.append(4)
                    l_vals_all.append(4)
        # a counter for the l_vals
        c1 = 0
        for j in three_line:
            x = []
            # the number of exponents and contraction coefficients
            n_line = int(lines[j].split()[1])
            c += 1
            # exponents
            exp = []
            # contraction coefficients
            coeff = []
            # now the loop for the exponents and coefficients
            for k in range(1, n_line + 1):
                # the line index
                n = j + k
                # append the exponents and coefficients
                exp.append(float(lines[n].split()[0]))
                coeff.append(float(lines[n].split()[1]))
            # the coordinates of the atom
            coords_init = [coords[h1][1], coords[h1][2], coords[h1][3]]
            # This is how we initialize an Atom data type for the coordinats to use in libint
            a = VECTOR(coords_init[0], coords_init[1], coords_init[2])
            # If the counter is the first one
            if c == 1:
                # initialize the shell by creating a shell type object
                shell = initialize_shell(int(l_vals[c1]), bool(purity[int(l_vals[c1])]), Py2Cpp_double(exp), Py2Cpp_double(coeff), a)
            else:
                # we now use the add_to_shell function to add the necessary values to the shell
                add_to_shell(shell, int(l_vals[c1]), bool(purity[int(l_vals[c1])]), Py2Cpp_double(exp), Py2Cpp_double(coeff), a)
            c1 += 1
        h1 += 1

    return shell, l_vals_all


def eigenvectors_molden(molden_filename: str, nbasis: int, l_vals: list):
    """
    This functions read the molecular orbitals eigenvectors and resort
    them so that we can use it with the libint.

    Args:

        molden_filename (string): The name of molden file.

        nbasis (integer): This is obtained from the liblibra_core.nbasis. It is necessary
                          since sometime the molden files will not append all the values
                          and there would be some values missing. It is better to use a
                          higher value of NDIGITS in the CP2K input file to have most of
                          the values.

        l_vals (list): All the angular momentum values return by the molden_file_to_libint_shell
                       function for the molden file.

    Returns:

        eigenvectors (list): All the sorted eigenvectors for use with the libint computed
                                    atomic orbital overlaps.

        orbital_energies (numpy array): All orbital energies for the eigenvectors.

    """
    # again reading the molden files and its lines
    file = open(molden_filename, 'r')
    lines = file.readlines()
    file.close()

    # energy lines
    orbital_energies = []
    ener_lines = []
    for i in range(len(lines)):
        if 'Ene='.lower() in lines[i].lower():
            ener_lines.append(i)
            # removing the 'Ene=' from the lines
            tmp = lines[i].lower().replace('ene=', '')
            # the energy value
            energy = float(tmp.split()[0])
            # appending the energy into orbital_energies
            orbital_energies.append(energy)
    # make it a numpy array
    orbital_energies = np.array(orbital_energies)

    # the file's per-l purity: cartesian shells resort directly to the
    # lexicographic canonical order the libint wrappers emit, and their
    # coefficients are rescaled to the unit-normalized component convention
    _purity = molden_purity(molden_filename, l_vals)
    _cscale = _cart_coeff_scale(l_vals, _purity)

    # loop-invariant: one resort map serves every MO
    new_indices = resort_eigenvectors(l_vals, _purity)
    # eigenvectors list
    eigenvectors = []
    # loop over the ener_lines
    for i in range(len(ener_lines)):
        # With this if there is no value in
        # the molden file it would automatically be zero
        eigenvector = np.zeros((nbasis))
        if i == len((ener_lines)) - 1:
            # end line to loop over for an energy value
            endl = len(lines)
        else:
            # end line to loop over for an energy value
            endl = ener_lines[i + 1]
        # now loop between them
        for j in range(ener_lines[i] + 3, endl):
            # the index in the molden file
            parts = lines[j].split()
            c = int(parts[0]) - 1
            # append the eigenvector value to
            # the np.zero eigenvector
            eigenvector[c] = float(parts[1])

#         eigenvector = np.array(eigenvector)
        # the new and sorted eigenvector
        eigenvector = eigenvector[new_indices] * _cscale
        # append it to the eigenvectors list
        eigenvectors.append(eigenvector)

    return eigenvectors, orbital_energies


def resort_eigenvectors(l_vals, purity=None):
    """
    This function resorts the indices of the eigenvectors based on the
    angular momentum values.

    Args:

        l_vals (list): All the values of the angular momentum for all the atoms.

        purity (dict): Optional {l: bool} spherical-purity map of the file
                       (molden_convention). None keeps the historical
                       all-spherical behavior; cartesian shells resort from
                       the molden cartesian order to the lexicographic
                       canonical order.

    Returns:

        new_indices (list): The new indices.

    """
    # new indices
    new_indices = []

    """
    The difference between the way MOLog and molden files
    store the eigenvectors is like this (as an example for the Cd atom):

    MOLog files        Molden files

    Cd  2s            Cd  2s
    Cd  3s            Cd  3s
    Cd  3py           Cd  3px
    Cd  3pz           Cd  3py
    Cd  3px           Cd  3pz
    Cd  4py           Cd  4px
    Cd  4pz           Cd  4py
    Cd  4px           Cd  4pz
    Cd  4d-2          Cd  4d0
    Cd  4d-1          Cd  4d+1
    Cd  4d0           Cd  4d-1
    Cd  4d+1          Cd  4d+2
    Cd  4d+2          Cd  4d-2
    Cd  5d-2          Cd  5d0
    Cd  5d-1          Cd  5d+1
    Cd  5d0           Cd  5d-1
    Cd  5d+1          Cd  5d+2
    Cd  5d+2          Cd  5d-2
    Cd  5f-3          Cd  5f0
    Cd  5f-2          Cd  5f+1
    Cd  5f-1          Cd  5f-1
    Cd  5f0           Cd  5f+2
    Cd  5f+1          Cd  5f-2
    Cd  5f+2          Cd  5f+3
    Cd  5f+3          Cd  5f-3
    """

    # setting up the counter
    c = 0
    # loop over all the angular momentum values
    for i in range(len(l_vals)):
        l_val = l_vals[i]
        # find the reorder indices needed for this l_val
        if purity is not None and l_val >= 1 and not purity.get(l_val, True):
            reordered_indices = np.array(_MOLDEN_CART_TO_LEX[l_val])
        else:
            reordered_indices = index_reorder(l_val)
        for j in range(len(reordered_indices)):
            # now append it by plus the counter since
            # we aim to do it for all the eigenvectors
            # and l values
            new_indices.append(c + reordered_indices[j])
        # increase the counter with t
        c += len(reordered_indices)

    # Return the new indices
    return new_indices


def index_reorder(l_val):
    """
    This function returns the reordered indices for an angular momentum value.

    Args:

        l_val (integer): The angular momentum value.

    Returns:

        new_order (numpy array): Contains the new indices.

    """
    # for s orbital
    if l_val == 0:
        new_order = [1]
    # for p orbital
    elif l_val == 1:
        new_order = [2, 3, 1]
    # for d orbital
    elif l_val == 2:
        new_order = [5, 3, 1, 2, 4]
    # for f orbital
    elif l_val == 3:
        new_order = [7, 5, 3, 1, 2, 4, 6]
    # for g orbital
    elif l_val == 4:
        new_order = [9, 7, 5, 3, 1, 2, 4, 6, 8]

    # The indeices
    return np.array(new_order) - 1


def write_molden_file(filename, sample_molden_file, path_to_trajectory, step):
    """
    This function writes a molden file for a geometry that contains the basis data. It doesn't write the
    eigenvectors since they are not needed anymore.
    Args:
        filename (strin): The molden file name to be written
        sample_molden_file (strin): The path to the sample molden file that contains the basis data
        path_to_trajectory (strin): The path to MD trajectory xyz file
        step (integer): The MD time step
    Returns:
        None
    """
    f = open(sample_molden_file)
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        if '[GTO]' in lines[i]:
            start = i
        elif lines[i].strip().upper().startswith(('[5D', '[7F', '[9G')):
            # every purity marker line survives, [9G] included
            end = i

    atoms, coords = data_read.read_atom_coords(path_to_trajectory, step)
    coords = coords * units.Angst
    f = open(filename, 'w')
    f.write('[Molden Format]\n [Atoms] AU\n')
    for i in range(len(coords)):
        f.write(F'{atoms[i]} {i+1} 0 {coords[i][0]} {coords[i][1]} {coords[i][2]}\n')
    for i in range(start, end + 1):
        f.write(lines[i])
    f.close()
