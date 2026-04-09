#!/usr/bin/env python3
# encoding: utf-8

"""Quarantine tests for the developer-only ``_save_debug_geometries``
helper.

The helper lives in :mod:`arc.job.adapters.ts.linear_test` and is
gated on the environment variable ``ARC_LINEAR_DEBUG_DUMP``: it is
inert by default and only writes Gaussian ``.gjf`` files into
``~/Desktop/xyz/linear/`` when an interactive debugging session opts
in.

These tests prove the helper stays inert in committed test runs.
They were extracted out of ``linear_test.py`` during the second
cleanup so the integration-test file does not also have to
host quarantine plumbing.
"""

import os
import unittest
from unittest.mock import patch


class TestDebugDumpQuarantine(unittest.TestCase):
    """``_save_debug_geometries`` is inert by default and only writes
    anything when explicitly opted in via ``ARC_LINEAR_DEBUG_DUMP``.
    """

    def _make_dummy_rxn_and_xyz(self):
        """Build a tiny rxn + xyz pair for the helper to receive.

        The helper does not care about chemistry — it just iterates
        ``rxn.r_species``, ``rxn.p_species`` and the supplied
        ``ts_xyzs`` list and feeds each one through ``save_geo``.
        """
        from arc.species import ARCSpecies
        from arc.reaction import ARCReaction
        sp_r = ARCSpecies(label='R', smiles='CC')
        sp_p = ARCSpecies(label='P', smiles='CC')
        rxn = ARCReaction(r_species=[sp_r], p_species=[sp_p])
        return rxn, sp_r.get_xyz()

    def test_debug_dump_is_inert_by_default(self):
        """Calling ``_save_debug_geometries`` without the env var set
        must not touch the filesystem.

        We assert this by patching ``save_geo`` to record any call;
        when the helper is correctly inert, no call should be made.
        """
        from arc.job.adapters.ts.linear_test import _save_debug_geometries
        rxn, ts_dummy = self._make_dummy_rxn_and_xyz()

        env_was = os.environ.pop('ARC_LINEAR_DEBUG_DUMP', None)
        try:
            with patch(
                    'arc.job.adapters.ts.linear_test.save_geo') as save_mock:
                _save_debug_geometries([ts_dummy], rxn)
            self.assertEqual(
                save_mock.call_count, 0,
                msg='_save_debug_geometries should be inert when '
                    'ARC_LINEAR_DEBUG_DUMP is unset, but save_geo was '
                    f'called {save_mock.call_count} times')
        finally:
            if env_was is not None:
                os.environ['ARC_LINEAR_DEBUG_DUMP'] = env_was

    def test_debug_dump_is_inert_when_env_is_zero_or_false(self):
        """Setting the env var to ``"0"``, ``"false"``, ``"False"`` or
        empty must also keep the helper inert."""
        from arc.job.adapters.ts.linear_test import _save_debug_geometries
        rxn, ts_dummy = self._make_dummy_rxn_and_xyz()

        env_was = os.environ.pop('ARC_LINEAR_DEBUG_DUMP', None)
        try:
            for value in ('0', '', 'false', 'False'):
                os.environ['ARC_LINEAR_DEBUG_DUMP'] = value
                with patch(
                        'arc.job.adapters.ts.linear_test.save_geo') as save_mock:
                    _save_debug_geometries([ts_dummy], rxn)
                self.assertEqual(
                    save_mock.call_count, 0,
                    msg=f'helper should be inert for env value {value!r}, '
                        f'but save_geo was called {save_mock.call_count} times')
        finally:
            os.environ.pop('ARC_LINEAR_DEBUG_DUMP', None)
            if env_was is not None:
                os.environ['ARC_LINEAR_DEBUG_DUMP'] = env_was


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
