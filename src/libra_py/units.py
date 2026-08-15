# *********************************************************************************
# * Copyright (C) 2018-2019 Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 2 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: units
   :platform: Unix, Windows
   :synopsis: This module defines the units conversion factors and implements the
       functions for unit conversion

.. moduleauthor:: Alexey V. Akimov

"""


# Unit conversions Parameters
# Length units
Angst = 1.8897261246257702                         # in Bohr (1.0/0.529177210903, CODATA 2018)

# Energy units (CODATA 2018 primaries; derived values are expressions
# of the primaries so they cannot drift on the next update)
inv_cm2ev = 1.0 / 8065.543937                      # 1 eV = 8065.543937 cm^-1 (CODATA 2018)
ev2Ha = 0.03674932217565499                        # (1.0/27.211386245988, CODATA 2018)
ev2au = 0.03674932217565499                        # (1.0/27.211386245988, CODATA 2018)
au2ev = 27.211386245988                            # 1 Ha = 27.211386245988 eV (CODATA 2018)
inv_cm2Ha = inv_cm2ev * ev2Ha
hartree = 627.509474                               # 1 Ha in kcal/mol (CODATA 2018)

# Time units
au2fs = 0.024188843265857                          # 1 a.u. of time in fs (CODATA 2018)
fs2au = 1.0 / au2fs
ps2au = fs2au * 1.0E+3
au2ps = au2fs * 1.0E-3

au2wavn = au2ev / inv_cm2ev                        # 219474.631... cm^-1 per Ha
wavn2au = 1.0 / au2wavn

# Mass units
amu = 1822.888486209                               # unified atomic mass unit (dalton) in electron masses (CODATA 2018)
m_e = 9.109383713928e-31                           # electron mass in kg

# Fundamental constants
boltzmann = 1.987204258641E-3                      # kB in kcal/(mol K) (CODATA 2018)
kB = 3.166811563E-6                                # kB in Ha/K (CODATA 2018)


def length_converter(inp_units, out_units):
    """Length units conversion factor

    Args:
        inp_units ( int ): defines the input units
        out_units ( int ): defines the output units

    Note:
        Both inp_units and out_units are encoded as follows:

            * 0 - atomic units (Bohrs)
            * 1 - Angstroms

    Returns:
        double: the length units conversion factor

    """

    scl = 1.0
    if inp_units == 0:  # input in Bohrs
        if out_units == 0:  # output in Bohrs
            scl = 1.0
        elif out_units == 1:  # output in Angstroms
            scl = 1.0 / Angst

    elif inp_units == 1:  # input in Angstroms
        if out_units == 0:  # output in Bohrs
            scl = Angst
        elif out_units == 1:  # output in Angstroms
            scl = 1.0

    return scl
