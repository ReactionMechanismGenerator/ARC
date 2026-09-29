"""
A module for approximate spin projection of a broken-symmetry wavefunction.
"""

import math
from typing import TYPE_CHECKING

from arc.common import get_logger
from arc.parser.parser import s_squared_expected_from_multiplicity

if TYPE_CHECKING:
    from arc.level import Level

logger = get_logger()

MIN_S2_SEPARATION = 0.1

MAX_PROJECTION_AMPLIFICATION = 2.0

BROKEN_SYMMETRY_S2_THRESHOLD = 1e-2


def _is_finite(value: float | None) -> bool:
    """
    Return whether a value is present and a finite number.

    Args:
        value (float | None): The value to test.

    Returns: bool
        ``True`` when the value is not ``None`` and is neither ``NaN`` nor infinite.
    """
    if value is None:
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _is_finite_s2(value: float | None) -> bool:
    """
    Return whether a value is usable as an ``<S**2>``.

    Args:
        value (float | None): The value to test.

    Returns: bool
        ``True`` when the value is present, finite and non-negative, which every
        expectation value of the total-spin operator is.
    """
    return _is_finite(value) and value >= 0.0


def target_low_spin_s_squared(multiplicity: int | float | None) -> float | None:
    """
    Return the spin-pure ``<S**2>`` of the low-spin state a projection targets.

    The value is ``S(S+1)`` of the target multiplicity, computed by
    ``arc.parser.parser.s_squared_expected_from_multiplicity``: 0 for a singlet,
    0.75 for a doublet, 2 for a triplet. A multiplicity that is missing, not a
    number, below one, not finite or not a whole number names no spin state, and is
    refused with a warning rather than treated as any particular one. A multiplicity
    is ``2S + 1`` for a total spin ``S`` that is a whole or half-integral number, so
    it is itself a positive integer and a fractional one names nothing: ``1.5`` would
    otherwise be read as ``S = 0.25``, which no state has.

    Args:
        multiplicity (int | float | None): The spin multiplicity of the target low-spin state.

    Returns: float | None
        The spin-pure ``S(S+1)`` of the target state, or ``None``.
    """
    s2_ls = s_squared_expected_from_multiplicity(multiplicity) \
        if _is_finite(multiplicity) and not isinstance(multiplicity, bool) and float(multiplicity).is_integer() \
        else None
    if not _is_finite_s2(s2_ls):
        logger.warning(f'Cannot spin-project onto a target low-spin state: {multiplicity!r} is not a spin '
                       f'multiplicity, so the spin-pure <S**2> that state is projected onto is undefined. '
                       f'The multiplicity of the state the projection targets is what decides that value.')
        return None
    return s2_ls


def yamaguchi_projected_energy(e_bs: float | None,
                               e_hs: float | None,
                               s2_bs: float | None,
                               s2_hs: float | None,
                               multiplicity: int | float | None,
                               ) -> float | None:
    """
    Compute the Yamaguchi approximate spin-projected low-spin energy.

    Applies the approximate spin projection (AP) scheme of Yamaguchi and co-workers,
    which removes the high-spin contamination of a broken-symmetry (BS) solution by
    extrapolating in ``<S**2>`` from the BS reference away from the high-spin (HS)
    reference, both computed at the same geometry and level::

        E_LS = E_BS + [(<S**2>_BS - <S**2>_LS) / (<S**2>_HS - <S**2>_BS)] * (E_BS - E_HS)

    ``<S**2>_LS = S_LS(S_LS + 1)`` is the spin-pure expectation value of the target
    low-spin state, and is taken from ``multiplicity``, the multiplicity of the state
    the projection targets: 0 for a singlet, 0.75 for a doublet, 2 for a triplet. The
    target state is an argument because it moves the answer by tens of kJ/mol, so there
    is no default. For a singlet target the expression reduces to the familiar closed
    form ``(<S**2>_HS * E_BS - <S**2>_BS * E_HS) / (<S**2>_HS - <S**2>_BS)``.

    K. Yamaguchi, F. Jensen, A. Dorigo, K. N. Houk, Chem. Phys. Lett. 1988, 149, 537.
    T. Soda et al., Chem. Phys. Lett. 2000, 319, 223 applies it to broken-symmetry DFT.

    Every refusal is logged as a warning and returns ``None`` rather than a number. The
    projection is refused when any argument is missing, non-finite, a negative ``<S**2>``
    or not a spin multiplicity; when the two references are separated in ``<S**2>`` by
    less than ``MIN_S2_SEPARATION``, below which they do not describe two distinguishable
    spin states; when the pair is inconsistent, meaning either that the HS reference is
    the less contaminated of the two or that the BS reference is less contaminated than
    the spin-pure target; and when the ratio multiplying ``E_BS - E_HS`` exceeds
    ``MAX_PROJECTION_AMPLIFICATION``.

    That ratio is ``w / (1 - w)`` in the high-spin weight ``w`` of the BS determinant, so
    it is the quantity that decides how far the projection moves the energy, and capping
    it rather than the denominator is what bounds the result. An ideal, fully spin-flipped
    broken-symmetry solution has ``w = 0.5`` and a ratio of exactly 1; a ratio above 1
    means the BS determinant carries more high-spin than target-spin character. The cap of
    ``MAX_PROJECTION_AMPLIFICATION`` admits ``w`` up to two thirds, a BS reference twice as
    high-spin as the ideal one, and refuses beyond it, where the correction exceeds twice
    the BS-to-HS energy gap and the pair no longer describes the target state. So the
    largest correction an accepted projection can apply is
    ``MAX_PROJECTION_AMPLIFICATION * |E_BS - E_HS|``, and the first pair accepted past the
    separation floor is bounded by the same amount as every other.

    A converged unrestricted determinant satisfies ``<S**2>_LS <= <S**2>_BS <= <S**2>_HS``,
    so an inconsistent ordering is a property of the calculation rather than of the chemistry.
    A broken-symmetry ``<S**2>`` below the spin-pure target by less than ``MIN_S2_SEPARATION``
    is the noise of a determinant that is spin-pure to within it, so the amplification it gives
    is taken as zero and the projected energy is ``E_BS`` itself. Amplifying by a negative
    number would return an energy ABOVE ``E_BS``, since ``E_BS - E_HS`` is negative, which is
    the wrong side of the reference the projection starts from.

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.
        multiplicity (int | float | None): The spin multiplicity of the target low-spin state.

    Returns: float | None
        The projected low-spin energy in the units of ``e_bs``, or ``None``.
    """
    s2_ls = target_low_spin_s_squared(multiplicity)
    if s2_ls is None:
        return None
    if not (_is_finite(e_bs) and _is_finite(e_hs)):
        return None
    if not (_is_finite_s2(s2_bs) and _is_finite_s2(s2_hs)):
        return None
    separation = s2_hs - s2_bs
    if separation < -MIN_S2_SEPARATION:
        logger.warning(f'Not projecting: the high-spin <S**2> ({s2_hs}) is below the broken-symmetry '
                       f'<S**2> ({s2_bs}), which a pair of converged unrestricted determinants of the '
                       f'same system cannot be. The two references do not describe the same calculation.')
        return None
    if separation < MIN_S2_SEPARATION:
        logger.warning(f'Not projecting: <S**2>_HS ({s2_hs}) and <S**2>_BS ({s2_bs}) are separated by '
                       f'{separation}, below the {MIN_S2_SEPARATION} two distinguishable spin states of '
                       f'the same system differ by.')
        return None
    if s2_bs < s2_ls - MIN_S2_SEPARATION:
        logger.warning(f'Not projecting: the broken-symmetry <S**2> ({s2_bs}) is below the spin-pure '
                       f'<S**2> of the target low-spin state ({s2_ls}). Either the target state or the '
                       f'broken-symmetry reference is not the one it is taken to be.')
        return None
    amplification = max((s2_bs - s2_ls) / separation, 0.0)
    if amplification > MAX_PROJECTION_AMPLIFICATION:
        logger.warning(f'Not projecting: with a broken-symmetry <S**2> of {s2_bs} against a spin-pure '
                       f'{s2_ls} and a high-spin {s2_hs}, the projection would move the energy by '
                       f'{amplification} times the broken-symmetry to high-spin gap, above the '
                       f'{MAX_PROJECTION_AMPLIFICATION} such an amplification of the two energies is '
                       f'trusted to. The broken-symmetry reference carries more high-spin than '
                       f'target-spin character.')
        return None
    return e_bs + amplification * (e_bs - e_hs)


def get_spin_projection(e_bs: float | None,
                        e_hs: float | None,
                        s2_bs: float | None,
                        s2_hs: float | None,
                        multiplicity: int | float | None,
                        level: 'Level | str | None',
                        xyz: dict | str | None,
                        e_restricted: float | None = None,
                        ) -> dict:
    """
    Assemble the record of an approximate spin projection.

    Collects the quantities the projection was computed from alongside its result, so the
    projected energy can be reproduced and audited from the record alone. ``r_u_gap`` is
    the restricted minus broken-symmetry energy difference, a diradicaloid diagnostic in
    its own right: it is positive by the variational principle whenever the broken-symmetry
    solution is genuinely lower, and near zero when the BS optimisation collapsed back onto
    the restricted solution. ``broken_symmetry`` reports whether the BS reference actually
    broke symmetry, judged by how far its ``<S**2>`` lies above the spin-pure ``s2_ls`` of
    the target state, and is ``None`` rather than ``False`` whenever that cannot be judged.

    ``level`` and ``xyz`` are the provenance of the energies: the single level of theory and
    the single geometry that ``e_bs``, ``e_hs`` and ``e_restricted`` were all computed at.
    The scheme extrapolates between two points of one potential energy surface, so energies
    taken from two levels or from each state's own optimized geometry are not a pair this
    projection is defined for, and the record names what it was given so a reader can tell.

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.
        multiplicity (int | float | None): The spin multiplicity of the target low-spin state.
        level (Level | str | None): The level of theory all the energies were computed at.
        xyz (dict | str | None): The geometry all the energies were computed at.
        e_restricted (float | None, optional): The restricted electronic energy of the same geometry.

    Returns: dict
        ``{'e_bs': float | None, 'e_hs': float | None, 's2_bs': float | None,
           's2_hs': float | None, 's2_ls': float | None, 'multiplicity': int | float | None,
           'level': str | None, 'xyz': dict | str | None, 'e_restricted': float | None,
           'r_u_gap': float | None, 'broken_symmetry': bool | None,
           'e_projected': float | None, 'scheme': 'yamaguchi_ap'}``.
    """
    s2_ls = target_low_spin_s_squared(multiplicity)
    r_u_gap = e_restricted - e_bs if _is_finite(e_restricted) and _is_finite(e_bs) else None
    broken_symmetry = s2_bs - s2_ls > BROKEN_SYMMETRY_S2_THRESHOLD \
        if _is_finite_s2(s2_bs) and _is_finite_s2(s2_ls) else None
    return {'e_bs': e_bs,
            'e_hs': e_hs,
            's2_bs': s2_bs,
            's2_hs': s2_hs,
            's2_ls': s2_ls,
            'multiplicity': multiplicity,
            'level': str(level) if level is not None else None,
            'xyz': xyz,
            'e_restricted': e_restricted,
            'r_u_gap': r_u_gap,
            'broken_symmetry': broken_symmetry,
            'e_projected': yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=s2_bs,
                                                      s2_hs=s2_hs, multiplicity=multiplicity)
                           if s2_ls is not None else None,
            'scheme': 'yamaguchi_ap',
            }
