"""The force rows a driver hands to the engine: the diagonal force terms -dE_i/dx per degree of freedom and state
(d1ham_adi), the rows of the states a driver does not compute NaN (the sentinel the engine's checks read - a zero
force is a force), and the hand-over of the engine's call-back after a change of the active state (hop_force_refresh)
shared by the CP2K and DFTB+ drivers."""
import numpy as np
from liblibra_core import CMATRIX, CMATRIXList
from libra_py import data_conv


class ForceHandover:
    """The object a driver returns to the engine's call-back: d1ham_adi and dc1_adi alone, or nothing"""
    pass


def fill_diagonal_forces(obj, diagonals):
    """d1ham_adi of a driver step from an (ndof, nstates) array of the diagonal force terms -dE_i/dx: one
    diagonal CMATRIX per degree of freedom through one bulk conversion each, in place of nstates element sets
    per degree of freedom, and dc1_adi as the zero matrices of the same size."""
    diagonals = np.asarray(diagonals, dtype=np.float64)
    ndof, nstates = diagonals.shape
    obj.d1ham_adi = CMATRIXList()
    obj.dc1_adi = CMATRIXList()
    for idof in range(ndof):
        obj.d1ham_adi.append(data_conv.nparray2CMATRIX(np.diag(diagonals[idof]).astype(np.complex128)))
        obj.dc1_adi.append(CMATRIX(nstates, nstates))


def force_diagonals(nstates, ndof, nat, forces, forces_all=None, active_state=None):
    """The (ndof, nstates) array of the diagonal force terms -dE_i/dx of a driver step, row 3 iat + k: every
    state's own forces from forces_all (nstates_all, nat, 3) when given - its first nstates states, a CP2K run
    may carry more excited states than are propagated - the forces (nat, 3) on the active state alone when
    active_state is given, the other states NaN (the sentinel of a row not handed over, which the engine's checks
    read - a zero force is a force), the same forces on every state otherwise. Rows beyond 3 nat stay zero."""
    diagonals = np.zeros((ndof, nstates))
    if forces_all is not None:
        forces_all = np.asarray(forces_all, dtype=np.float64)
        if forces_all.shape[0] < nstates:
            raise ValueError(F"per-state forces for {forces_all.shape[0]} states cannot fill {nstates} propagated "
                             F"states")
        diagonals[:3 * nat] = -forces_all[:nstates].transpose(1, 2, 0).reshape(3 * nat, nstates)
    elif active_state is not None:
        diagonals[:3 * nat] = np.nan
        diagonals[:3 * nat, active_state] = -np.asarray(forces, dtype=np.float64).reshape(3 * nat)
    else:
        diagonals[:3 * nat] = -np.asarray(forces, dtype=np.float64).reshape(3 * nat, 1)
    return diagonals


def refreshed_rows(params, itraj, nat, force_of):
    """The hand-over of the engine's call-back after a change of the active state (hop_force_refresh:
    params["force_refresh"] = {trajectory: new active state}): the rows handed over at the trajectory's regular
    call of the step (params["force_rows"]), the new state's row computed by force_of(new_state) - the forces
    (nat, 3) on that state at the same geometry - when the row is NaN (not handed over), nothing else. A
    trajectory absent from the call-back hands back an empty object, a row the regular call filled (the ground
    state, force_source all_states or gs, a Niehaus FRC.DAT) hands the rows back as they are."""
    obj = ForceHandover()
    refresh = params["force_refresh"]
    if itraj not in refresh:
        return obj
    if itraj not in params.get("force_rows", {}):
        raise RuntimeError(F"force_refresh for trajectory {itraj} before its regular call of the step")
    rows = np.array(params["force_rows"][itraj], dtype=np.float64)
    new_state = int(refresh[itraj])
    if np.isnan(rows[:3 * nat, new_state]).any():
        rows[:3 * nat, new_state] = -np.asarray(force_of(new_state), dtype=np.float64).reshape(3 * nat)
        params["force_rows"][itraj] = rows
    fill_diagonal_forces(obj, rows)
    return obj
