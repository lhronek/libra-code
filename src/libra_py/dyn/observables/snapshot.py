"""
Compact observable snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from libra_py.params_check import refuse_unknown_keys

from .energies import (
    active_potential_energy,
    ehrenfest_potential_energy,
    kinetic_energy,
    total_energy,
)
from .populations import (
    active_state_counts,
    amplitude_populations,
    density_populations,
    mean_populations,
)


LEGACY_OUTPUT_LEVEL_KEYWORDS = {
    1: (
        "timestep",
        "time",
        "Ekin_ave",
        "Epot_ave",
        "Etot_ave",
        "dEkin_ave",
        "dEpot_ave",
        "dEtot_ave",
        "Etherm",
        "E_NHC",
        "tcnbra_ekin",
        "tcnbra_thermostat_energy",
        "ekin_aux_var",
    ),
    2: (
        "states",
        "states_dia",
        "se_pop_adi",
        "se_pop_dia",
        "sh_pop_adi",
        "sh_pop_dia",
        "sh_pop_adi_TR",
        "sh_pop_dia_TR",
        "mash_pop_adi",
        "mash_pop_dia",
        "fssh3_average_errors",
        "y_aux_var",
        "p_aux_var",
        "f_aux_var",
    ),
    3: (
        "D_adi",
        "D_dia",
        "coherence_adi",
        "coherence_dia",
        "q",
        "p",
        "f",
        "Cadi",
        "Cdia",
        "q_mm",
        "p_mm",
        "wp_width",
        "p_quant",
        "VP",
        "f_xf",
        "qtsh_f_nc",
        "ave_decoherence_rates",
    ),
    4: (
        "hvib_adi",
        "hvib_dia",
        "St",
        "basis_transform",
        "projector",
        "q_aux",
        "p_aux",
        "nab_phase",
        "energy_gaps",
        "mean_energy_gaps",
        "energy_gaps2",
        "mean_energy_gaps2",
        "energy_gap_fluctuations",
        "energy_gap_correlations",
    ),
    5: ("dc1_adi",),
}

# every old-style keyword the snapshot serves - a requested name outside it is refused with the nearest known one
LEGACY_KEYWORDS = frozenset(name for names in LEGACY_OUTPUT_LEVEL_KEYWORDS.values() for name in names)

# the five dead names of dynamics/tsh/save.DEAD_PROPERTIES (the certifier pins the equality), kept here so the package stays
# free of liblibra - the live datasets are the ones in each step's eigenvector labels, the tracked labels of 2022 are not kept
LEGACY_DEAD_KEYWORDS = {
    "SH_pop": "sh_pop_adi holds the surface-hopping population in each step's eigenvector labels",
    "SH_pop_raw": "sh_pop_adi holds the surface-hopping population in the eigenvector labels",
    "D_adi_raw": "D_adi holds the adiabatic density in the eigenvector labels",
    "D_dia_raw": "D_dia holds the diabatic density - the diabatic labels are never relabelled",
    "Ekin_ave_qtsh": "no observable - the QTSH kinematic momentum and its kinetic energy left the engine in 2024",
}


def check_legacy_keywords(keywords) -> None:
    """Refuse a requested old-style keyword no observable serves: the five legacy names with what replaces them, any
    other unknown name with the nearest known one."""
    dead = [name for name in keywords if name in LEGACY_DEAD_KEYWORDS]
    if dead:
        raise ValueError("the observable keywords name " + ", ".join(F"{n} ({LEGACY_DEAD_KEYWORDS[n]})" for n in dead))
    refuse_unknown_keys(dict.fromkeys(keywords), LEGACY_KEYWORDS, "compute_legacy_observables", "keywords",
                        "no observable serves")


@dataclass(frozen=True)
class ObservableConfig:
    """Select which observables to compute for a snapshot."""

    rep: str = "adiabatic"
    keywords: tuple[str, ...] | None = None
    output_level: int | None = None
    populations: bool = True
    density_populations: bool = False
    active_counts: bool = True
    energies: bool = True
    potential: str = "active"
    include_coordinates: bool = False
    include_momenta: bool = False
    include_forces: bool = False
    include_amplitudes: bool = False


def compute_observables(
    storage: Any,
    traj: Any,
    config: ObservableConfig | None = None,
    step: int | None = None,
    time: float | None = None,
) -> dict[str, Any]:
    """
    Compute a compact dict of requested observables.

    The function only derives quantities from tensors already present in
    TensorStorage. Large raw tensors are included only when requested.
    """

    config = config or ObservableConfig()
    keywords = observable_keywords(config)
    if keywords is not None:
        return compute_legacy_observables(
            storage,
            traj,
            keywords,
            rep=config.rep,
            potential=config.potential,
            step=step,
            time=time,
        )

    idx = traj.tbf_ids
    result: dict[str, Any] = {
        "traj_id": traj.id,
        "tbf_ids": list(idx),
    }
    if step is not None:
        result["step"] = step
    if time is not None:
        result["time"] = time

    if config.populations:
        result["populations"] = amplitude_populations(storage, traj, config.rep)
        result["mean_populations"] = mean_populations(storage, traj, config.rep)
    if config.density_populations:
        result["density_populations"] = density_populations(storage, traj, config.rep)
    if config.active_counts:
        result["active_state_counts"] = active_state_counts(storage, traj, config.rep)
    if config.energies:
        result["kinetic_energy"] = kinetic_energy(storage, traj)
        if config.potential == "active":
            result["potential_energy"] = active_potential_energy(storage, traj, config.rep)
        elif config.potential == "ehrenfest":
            result["potential_energy"] = ehrenfest_potential_energy(storage, traj, config.rep)
        else:
            raise ValueError("potential must be 'active' or 'ehrenfest'")
        result["total_energy"] = total_energy(storage, traj, config.rep, config.potential)

    if config.include_coordinates:
        result["q"] = storage.q[traj.id, idx]
    if config.include_momenta:
        result["p"] = storage.p[traj.id, idx]
    if config.include_forces:
        result["f"] = storage.f[traj.id, idx]
    if config.include_amplitudes:
        result["amplitudes"] = (
            storage.ampl_adi[traj.id, idx]
            if config.rep == "adiabatic"
            else storage.ampl_dia[traj.id, idx]
        )

    return result


def observable_keywords(config: ObservableConfig) -> tuple[str, ...] | None:
    """Return explicit old-style observable keywords selected by config."""

    if config.keywords is not None:
        return tuple(config.keywords)
    if config.output_level is None:
        return None
    return legacy_observable_keywords(config.output_level)


def legacy_observable_keywords(output_level: int) -> tuple[str, ...]:
    """Return the old `dynamics/tsh/save.py` keyword set up to an output level."""

    if output_level < 0:
        raise ValueError("output_level must be non-negative")
    names: list[str] = []
    for level in sorted(LEGACY_OUTPUT_LEVEL_KEYWORDS):
        if output_level >= level:
            names.extend(LEGACY_OUTPUT_LEVEL_KEYWORDS[level])
    return tuple(names)


def compute_legacy_observables(
    storage: Any,
    traj: Any,
    keywords: Iterable[str],
    rep: str = "adiabatic",
    potential: str = "active",
    step: int | None = None,
    time: float | None = None,
) -> dict[str, Any]:
    """
    Compute old-save-style observable names for one trajectory slice.

    A keyword no observable serves is refused (check_legacy_keywords). A served method-specific key is omitted when
    the corresponding TensorStorage field does not exist. This keeps snapshots economical and avoids fabricating
    values for methods that are not active.
    """

    keywords = tuple(keywords)
    check_legacy_keywords(keywords)
    requested = set(keywords)
    idx = traj.tbf_ids
    result: dict[str, Any] = {
        "traj_id": traj.id,
        "tbf_ids": list(idx),
    }

    if "timestep" in requested:
        result["timestep"] = storage.timestep if step is None else step
    if "time" in requested and time is not None:
        result["time"] = time

    ekin = kinetic_energy(storage, traj)
    if potential == "active":
        epot = active_potential_energy(storage, traj, rep)
    elif potential == "ehrenfest":
        epot = ehrenfest_potential_energy(storage, traj, rep)
    else:
        raise ValueError("potential must be 'active' or 'ehrenfest'")
    etot = ekin + epot
    _add_average_triplet(result, requested, "Ekin", ekin)
    _add_average_triplet(result, requested, "Epot", epot)
    _add_average_triplet(result, requested, "Etot", etot)

    _add_optional_scalar(result, requested, "Etherm", storage, 0.0)
    _add_optional_scalar(result, requested, "E_NHC", storage, 0.0)
    _add_optional_array(result, requested, "tcnbra_ekin", storage)
    _add_optional_array(result, requested, "tcnbra_thermostat_energy", storage)
    _add_optional_array(result, requested, "ekin_aux_var", storage)
    _add_active_states(result, requested, storage, traj)
    _add_population_observables(result, requested, storage, traj)
    _add_density_observables(result, requested, storage, traj)
    _add_raw_field_observables(result, requested, storage, traj)
    _add_gap_observables(result, requested, storage, traj)
    return result


def _add_average_triplet(result, requested, prefix, values) -> None:
    values = np.asarray(values)
    if f"{prefix}_ave" in requested:
        result[f"{prefix}_ave"] = float(np.mean(values))
    if f"d{prefix}_ave" in requested:
        result[f"d{prefix}_ave"] = float(np.std(values))


def _add_optional_scalar(result, requested, name, storage, default) -> None:
    if name not in requested:
        return
    value = getattr(storage, name, default)
    result[name] = _compact_optional_value(value, default)


def _warn_absent(name, field) -> None:
    """A requested keyword whose storage field is absent is omitted - said once per call, not silently"""
    print(F"WARNING in compute_legacy_observables(): the keyword {name} was requested but the storage holds no {field} - omitted")


def _add_optional_array(result, requested, name, storage) -> None:
    if name not in requested:
        return
    value = getattr(storage, name, None)
    if value is None:
        _warn_absent(name, name)
        return
    result[name] = np.asarray(value)


def _add_active_states(result, requested, storage, traj) -> None:
    idx = traj.tbf_ids
    if "states" in requested:
        result["states"] = storage.act_states[traj.id, idx]
    if "states_dia" in requested:
        result["states_dia"] = storage.act_states_dia[traj.id, idx]


def _add_population_observables(result, requested, storage, traj) -> None:
    if "se_pop_adi" in requested:
        result["se_pop_adi"] = mean_populations(storage, traj, "adiabatic")
    if "se_pop_dia" in requested:
        result["se_pop_dia"] = mean_populations(storage, traj, "diabatic")

    if "sh_pop_adi" in requested:
        result["sh_pop_adi"] = _normalized_counts(storage, traj, "adiabatic")
    if "sh_pop_dia" in requested:
        result["sh_pop_dia"] = _normalized_counts(storage, traj, "diabatic")

    for source, target in (
        ("sh_pop_adi", "sh_pop_adi_TR"),
        ("sh_pop_dia", "sh_pop_dia_TR"),
        ("sh_pop_adi", "mash_pop_adi"),
        ("sh_pop_dia", "mash_pop_dia"),
    ):
        if target in requested:
            result[target] = result.get(source, _normalized_counts(
                storage,
                traj,
                "adiabatic" if source.endswith("adi") else "diabatic",
            ))


def _add_density_observables(result, requested, storage, traj) -> None:
    if "D_adi" in requested:
        result["D_adi"] = _average_density(storage, traj, "adiabatic")
    if "D_dia" in requested:
        result["D_dia"] = _average_density(storage, traj, "diabatic")
    if "coherence_adi" in requested:
        result["coherence_adi"] = _coherence_indicator(_average_density(storage, traj, "adiabatic"))
    if "coherence_dia" in requested:
        result["coherence_dia"] = _coherence_indicator(_average_density(storage, traj, "diabatic"))


def _add_raw_field_observables(result, requested, storage, traj) -> None:
    field_map = {
        "q": "q",
        "p": "p",
        "f": "f",
        "Cadi": "ampl_adi",
        "Cdia": "ampl_dia",
        "q_mm": "q_mm",
        "p_mm": "p_mm",
        "wp_width": "wp_width",
        "p_quant": "p_quant",
        "VP": "VP",
        "f_xf": "f_xf",
        "qtsh_f_nc": "qtsh_f_nc",
        "ave_decoherence_rates": "ave_decoherence_rates",
        "hvib_adi": "hvib_adi",
        "hvib_dia": "hvib_dia",
        "St": "time_overlap_adi",
        "basis_transform": "basis_transform",
        "projector": "proj_adi",
        "q_aux": "q_aux",
        "p_aux": "p_aux",
        "nab_phase": "nab_phase",
        "dc1_adi": "dc1_adi",
        "fssh3_average_errors": "fssh3_errors",
        "y_aux_var": "y_aux_var",
        "p_aux_var": "p_aux_var",
        "f_aux_var": "f_aux_var",
    }
    idx = traj.tbf_ids
    for name, field in field_map.items():
        if name not in requested:
            continue
        value = getattr(storage, field, None)
        if value is None:
            _warn_absent(name, field)
            continue
        result[name] = value[traj.id, idx]


def _add_gap_observables(result, requested, storage, traj) -> None:
    gap_fields = {
        "energy_gaps": "gaps_curr",
        "mean_energy_gaps": "mean_gap",
        "energy_gaps2": "gaps_curr",
        "mean_energy_gaps2": "mean_gap2",
        "energy_gap_fluctuations": "gap_fluctuations",
        "energy_gap_correlations": "gap_correlations",
    }
    idx = traj.tbf_ids
    for name, field in gap_fields.items():
        if name not in requested:
            continue
        value = getattr(storage, field, None)
        if value is not None:
            selected = np.asarray(value[traj.id, idx])
            result[name] = selected * selected if name == "energy_gaps2" else selected
        elif name in ("energy_gaps", "energy_gaps2"):
            gaps = _hamiltonian_energy_gaps(storage, traj)
            result[name] = gaps * gaps if name == "energy_gaps2" else gaps
        else:
            _warn_absent(name, field)


def _normalized_counts(storage, traj, rep):
    counts = active_state_counts(storage, traj, rep).astype(float)
    total = np.sum(counts)
    return counts / total if total > 0 else counts


def _average_density(storage, traj, rep):
    idx = traj.tbf_ids
    if rep == "adiabatic":
        density = np.asarray(storage.dm_adi[traj.id, idx])
        amplitudes = np.asarray(storage.ampl_adi[traj.id, idx])
    elif rep == "diabatic":
        density = np.asarray(storage.dm_dia[traj.id, idx])
        amplitudes = np.asarray(storage.ampl_dia[traj.id, idx])
    else:
        raise ValueError("rep must be 'adiabatic' or 'diabatic'")

    if np.any(density):
        return np.mean(density, axis=0)
    built = np.einsum("...i,...j->...ij", amplitudes, np.conjugate(amplitudes))
    return np.mean(built, axis=0)


def _coherence_indicator(density):
    pops = np.real(np.diag(density))
    denom = np.sqrt(np.outer(pops, pops))
    out = np.zeros_like(np.real(density), dtype=float)
    np.divide(np.abs(density), denom, out=out, where=denom > 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def _hamiltonian_energy_gaps(storage, traj):
    idx = traj.tbf_ids
    energies = np.real(np.diagonal(storage.ham_adi[traj.id, idx], axis1=-2, axis2=-1))
    return energies[..., :, None] - energies[..., None, :]


def _compact_optional_value(value, default):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr
