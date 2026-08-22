"""
A module for approximate spin projection of a broken-symmetry wavefunction.
"""

import math

from arc.common import get_logger

logger = get_logger()

MIN_S2_SEPARATION = 0.1

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


def yamaguchi_projected_energy(e_bs: float | None,
                               e_hs: float | None,
                               s2_bs: float | None,
                               s2_hs: float | None,
                               s2_ls: float | None = 0.0,
                               ) -> float | None:
    """
    Compute the Yamaguchi approximate spin-projected low-spin energy.

    Applies the approximate spin projection (AP) scheme of Yamaguchi and co-workers,
    which removes the high-spin contamination of a broken-symmetry (BS) solution by
    extrapolating in ``<S**2>`` from the BS reference away from the high-spin (HS)
    reference, both computed at the same geometry and level::

        E_LS = E_BS + [(<S**2>_BS - <S**2>_LS) / (<S**2>_HS - <S**2>_BS)] * (E_BS - E_HS)

    ``<S**2>_LS = S_LS(S_LS + 1)`` is the spin-pure expectation value of the target
    low-spin state, ``0`` for a singlet (the default), ``0.75`` for a doublet, ``2``
    for a triplet. For a singlet target the expression reduces to the familiar closed
    form ``(<S**2>_HS * E_BS - <S**2>_BS * E_HS) / (<S**2>_HS - <S**2>_BS)``.

    K. Yamaguchi, F. Jensen, A. Dorigo, K. N. Houk, Chem. Phys. Lett. 1988, 149, 537.
    T. Soda et al., Chem. Phys. Lett. 2000, 319, 223 applies it to broken-symmetry DFT.

    Returns ``None`` rather than a number whenever the projection is not defined: when
    any argument is missing, non-finite, or a negative ``<S**2>``; when the two
    references are separated in ``<S**2>`` by less than ``MIN_S2_SEPARATION``, below
    which the ratio multiplying ``E_BS - E_HS`` exceeds ten and the result is dominated
    by the error on the two energies; or when the pair is inconsistent, meaning either
    that the HS reference is the less contaminated of the two or that the BS reference
    is less contaminated than the spin-pure target. A converged unrestricted determinant
    satisfies ``<S**2>_LS <= <S**2>_BS <= <S**2>_HS``, so an inconsistent ordering is a
    property of the calculation rather than of the chemistry and is logged as a warning.

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.
        s2_ls (float | None, optional): The spin-pure ``S(S+1)`` of the target low-spin
                                        state. Defaults to ``0.0``, a singlet.

    Returns: float | None
        The projected low-spin energy in the units of ``e_bs``, or ``None``.
    """
    if not (_is_finite(e_bs) and _is_finite(e_hs)):
        return None
    if not (_is_finite_s2(s2_bs) and _is_finite_s2(s2_hs) and _is_finite_s2(s2_ls)):
        return None
    separation = s2_hs - s2_bs
    if separation < -MIN_S2_SEPARATION:
        logger.warning(f'Not projecting: the high-spin <S**2> ({s2_hs}) is below the broken-symmetry '
                       f'<S**2> ({s2_bs}), which a pair of converged unrestricted determinants of the '
                       f'same system cannot be. The two references do not describe the same calculation.')
        return None
    if separation < MIN_S2_SEPARATION:
        logger.debug(f'Not projecting: <S**2>_HS ({s2_hs}) and <S**2>_BS ({s2_bs}) are separated '
                     f'by {separation}, below the {MIN_S2_SEPARATION} required.')
        return None
    if s2_bs < s2_ls - MIN_S2_SEPARATION:
        logger.warning(f'Not projecting: the broken-symmetry <S**2> ({s2_bs}) is below the spin-pure '
                       f'<S**2> of the target low-spin state ({s2_ls}). Either the target state or the '
                       f'broken-symmetry reference is not the one it is taken to be.')
        return None
    return e_bs + (s2_bs - s2_ls) / separation * (e_bs - e_hs)


def get_spin_projection(e_bs: float | None,
                        e_hs: float | None,
                        s2_bs: float | None,
                        s2_hs: float | None,
                        e_restricted: float | None = None,
                        s2_ls: float | None = 0.0,
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

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.
        e_restricted (float | None, optional): The restricted electronic energy of the same geometry.
        s2_ls (float | None, optional): The spin-pure ``S(S+1)`` of the target low-spin
                                        state. Defaults to ``0.0``, a singlet.

    Returns: dict
        ``{'e_bs': float | None, 'e_hs': float | None, 's2_bs': float | None,
           's2_hs': float | None, 's2_ls': float | None, 'e_restricted': float | None,
           'r_u_gap': float | None, 'broken_symmetry': bool | None,
           'e_projected': float | None, 'scheme': 'yamaguchi_ap'}``.
    """
    r_u_gap = e_restricted - e_bs if _is_finite(e_restricted) and _is_finite(e_bs) else None
    broken_symmetry = s2_bs - s2_ls > BROKEN_SYMMETRY_S2_THRESHOLD \
        if _is_finite_s2(s2_bs) and _is_finite_s2(s2_ls) else None
    return {'e_bs': e_bs,
            'e_hs': e_hs,
            's2_bs': s2_bs,
            's2_hs': s2_hs,
            's2_ls': s2_ls,
            'e_restricted': e_restricted,
            'r_u_gap': r_u_gap,
            'broken_symmetry': broken_symmetry,
            'e_projected': yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=s2_bs,
                                                      s2_hs=s2_hs, s2_ls=s2_ls),
            'scheme': 'yamaguchi_ap',
            }
