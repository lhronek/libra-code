# *********************************************************************************
# * Copyright (C) 2025 Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# ***********************************************************************************
"""
.. module:: Martens_for_exact_pytorch
   :platform: Unix, Windows
   :synopsis: This module implements 2 Martens models written in PyTorch way for the use
              with the exact DVR calculations. This format is compatible with Libra's DVR
              implementation
.. moduleauthor:: Alexey V. Akimov

"""

import torch
from libra_py.params_check import refuse_unknown_keys, MODEL_CALL_KEYS


# the keys Martens_model reads from its params dict (its defaults and critical keys included) - refused at the entry, the dispatch layer's keys skipped
MARTENS_MODEL_KEYS = frozenset((
    "Va", "Vb", "Vc"
))


def sech(x):
  return 1 / torch.cosh(x)

def Martens_model(q, params):
    """
    q - Tensor(ndof)

    Martens_model1 is just this one but with Vc = 0.0
    Martens_model2 is              --        Vc = 0.4
    """
    refuse_unknown_keys(params, MARTENS_MODEL_KEYS, "Martens_for_exact_pytorch.Martens_model", "params", "Martens_model does not read", skip=MODEL_CALL_KEYS)
    #params = {"Va": 0.00625, "Vb": 0.0106}
    Va = params.get("Va", 0.00625)
    Vb = params.get("Vb", 0.0106)
    Vc = params.get("Vc", 0.0)
    x = q[0]
    y = q[1]

    return Va * (sech(2.0*x))**2 + 0.5 * Vb * (y + Vc * (x**2 - 1.0 ) )**2
