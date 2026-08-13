"""
A module for approximate spin projection of a broken-symmetry wavefunction.
"""

from arc.common import get_logger

logger = get_logger()

# The smallest <S**2> separation between the broken-symmetry and high-spin
# references for which the Yamaguchi projection is evaluated.
MIN_S2_SEPARATION = 1e-3

# The <S**2> below which a broken-symmetry reference is reported as having
# collapsed onto the closed-shell solution.
BROKEN_SYMMETRY_S2_THRESHOLD = 1e-2


def yamaguchi_projected_energy(e_bs: float | None,
                               e_hs: float | None,
                               s2_bs: float | None,
                               s2_hs: float | None,
                               ) -> float | None:
    """
    Compute the Yamaguchi approximate spin-projected low-spin energy.

    Applies the approximate spin projection (AP) scheme of Yamaguchi and co-workers,
    which removes the high-spin contamination of a broken-symmetry (BS) solution by
    interpolating in ``<S**2>`` between the BS reference and the high-spin (HS)
    reference computed at the same geometry and level::

        E_LS = (<S**2>_HS * E_BS - <S**2>_BS * E_HS) / (<S**2>_HS - <S**2>_BS)

    K. Yamaguchi, F. Jensen, A. Dorigo, K. N. Houk, Chem. Phys. Lett. 1988, 149, 537.
    T. Soda et al., Chem. Phys. Lett. 2000, 319, 223 applies it to broken-symmetry DFT.

    Returns ``None`` rather than an extrapolated value whenever the interpolation is not
    defined: when any argument is missing, when the two references are not separated in
    ``<S**2>`` by at least ``MIN_S2_SEPARATION``, or when the BS reference is the more
    spin-contaminated of the two, which places the low-spin state outside the interval
    the two references bracket.

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.

    Returns: float | None
        The projected low-spin energy in the units of ``e_bs``, or ``None``.
    """
    if any(value is None for value in [e_bs, e_hs, s2_bs, s2_hs]):
        return None
    separation = s2_hs - s2_bs
    if separation < MIN_S2_SEPARATION:
        logger.debug(f'Not projecting: <S**2>_HS ({s2_hs}) and <S**2>_BS ({s2_bs}) are separated '
                     f'by {separation}, below the {MIN_S2_SEPARATION} required.')
        return None
    return (s2_hs * e_bs - s2_bs * e_hs) / separation


def get_spin_projection(e_bs: float | None,
                        e_hs: float | None,
                        s2_bs: float | None,
                        s2_hs: float | None,
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
    broke symmetry, judged by its ``<S**2>`` exceeding ``BROKEN_SYMMETRY_S2_THRESHOLD``.

    Args:
        e_bs (float | None): The broken-symmetry electronic energy.
        e_hs (float | None): The high-spin electronic energy, at the same geometry and level.
        s2_bs (float | None): The broken-symmetry ``<S**2>``.
        s2_hs (float | None): The high-spin ``<S**2>``.
        e_restricted (float | None, optional): The restricted electronic energy of the same geometry.

    Returns: dict
        ``{'e_bs': float | None, 'e_hs': float | None, 's2_bs': float | None,
           's2_hs': float | None, 'e_restricted': float | None, 'r_u_gap': float | None,
           'broken_symmetry': bool | None, 'e_projected': float | None,
           'scheme': 'yamaguchi_ap'}``.
    """
    return {'e_bs': e_bs,
            'e_hs': e_hs,
            's2_bs': s2_bs,
            's2_hs': s2_hs,
            'e_restricted': e_restricted,
            'r_u_gap': e_restricted - e_bs if e_restricted is not None and e_bs is not None else None,
            'broken_symmetry': s2_bs > BROKEN_SYMMETRY_S2_THRESHOLD if s2_bs is not None else None,
            'e_projected': yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=s2_bs, s2_hs=s2_hs),
            'scheme': 'yamaguchi_ap',
            }
