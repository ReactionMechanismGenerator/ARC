#!/usr/bin/env python3
# encoding: utf-8

"""Tests for the TCKDB upload-sweep dispatch, focused on the
computed-reaction path's partial-reaction species salvage.

These exercise ``_run_reaction_sweep`` directly with a stub adapter so
no network / real TCKDB client is involved.
"""

import copy
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass

from arc.tckdb.adapter_test import _reaction_output_doc
from arc.tckdb.sweep import _run_reaction_sweep


@dataclass
class _Outcome:
    """Minimal stand-in for adapter.UploadOutcome (only ``status`` read)."""

    status: str
    error: str | None = None


class _StubAdapter:
    """Records which submit_* methods the sweep called and with what."""

    def __init__(self, *, species_status='uploaded'):
        self.reaction_calls = []  # list[(label, is_partial)]
        self.species_calls = []  # list[label]
        self._species_status = species_status

    def submit_computed_reaction_from_output(self, *, output_doc, reaction_record, is_partial=False):
        self.reaction_calls.append((reaction_record.get('label'), is_partial))
        # Mirror the real adapter: partial reactions are sidecar-only
        # (status "skipped"), converged reactions upload.
        return _Outcome(status='skipped' if is_partial else 'uploaded')

    def submit_computed_species_from_output(self, *, output_doc, species_record):
        self.species_calls.append(species_record.get('label'))
        return _Outcome(status=self._species_status)


class _StubConfig:
    """Just the field ``_run_reaction_sweep`` reads."""

    def __init__(self, *, allow_partial_uploads=True):
        self.allow_partial_uploads = allow_partial_uploads


def _doc_with(*, ts_converged, species_converged=None):
    """Reaction output doc with per-record convergence flags set.

    ``species_converged`` maps species label -> bool; unlisted species
    default to converged. The TS's ``converged`` is set from
    ``ts_converged``.
    """
    doc = copy.deepcopy(_reaction_output_doc())
    species_converged = species_converged or {}
    for s in doc['species']:
        s['converged'] = species_converged.get(s['label'], True)
    for ts in doc['transition_states']:
        ts['converged'] = ts_converged
    return doc


class TestPartialReactionSpeciesSalvage(unittest.TestCase):
    def _run(self, doc, config):
        adapter = _StubAdapter()
        with redirect_stdout(io.StringIO()) as out:
            _run_reaction_sweep(adapter=adapter, output_doc=doc, tckdb_config=config)
        return adapter, out.getvalue()

    def test_converged_reaction_uploads_bundle_no_species_salvage(self):
        # Baseline: a fully converged reaction uploads the bundle inline;
        # species are NOT separately salvaged (no double upload).
        doc = _doc_with(ts_converged=True)
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=True))
        self.assertEqual(adapter.reaction_calls, [('CHO + CH4 <=> CH2O + CH3', False)])
        self.assertEqual(adapter.species_calls, [])

    def test_partial_reaction_salvages_all_converged_species(self):
        # TS non-converged, all four minima converged, partials allowed:
        # reaction goes sidecar-only (is_partial=True) AND every converged
        # reactant/product species is salvaged via the computed-species path.
        doc = _doc_with(ts_converged=False)
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=True))
        self.assertEqual(adapter.reaction_calls, [('CHO + CH4 <=> CH2O + CH3', True)])
        self.assertEqual(
            sorted(adapter.species_calls), ['CH2O', 'CH3', 'CH4', 'CHO'],
        )

    def test_partial_species_salvaged_even_when_partials_disabled(self):
        # allow_partial_uploads=False governs only the reaction/kinetics
        # record: the reaction is skipped (no reaction call), but the
        # converged species are STILL salvaged.
        doc = _doc_with(ts_converged=False)
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=False))
        self.assertEqual(adapter.reaction_calls, [])
        self.assertEqual(
            sorted(adapter.species_calls), ['CH2O', 'CH3', 'CH4', 'CHO'],
        )

    def test_non_converged_species_not_uploaded(self):
        # One product (CH2O) did not converge → it is skipped; the other
        # three converged species are still uploaded.
        doc = _doc_with(ts_converged=False, species_converged={'CH2O': False})
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=True))
        self.assertNotIn('CH2O', adapter.species_calls)
        self.assertEqual(sorted(adapter.species_calls), ['CH3', 'CH4', 'CHO'])

    def test_species_shared_across_partial_reactions_uploaded_once(self):
        # Two partial reactions share reactant CHO; it must be uploaded
        # only once (dedup by label across the sweep).
        doc = _doc_with(ts_converged=False)
        rxn2 = copy.deepcopy(doc['reactions'][0])
        rxn2['label'] = 'CHO + CH3 <=> CH2O + CH4'
        rxn2['reactant_labels'] = ['CHO', 'CH3']
        rxn2['product_labels'] = ['CH2O', 'CH4']
        rxn2['ts_label'] = None
        doc['reactions'].append(rxn2)
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=True))
        # Each of the four species uploaded exactly once, no duplicates.
        self.assertEqual(sorted(adapter.species_calls), ['CH2O', 'CH3', 'CH4', 'CHO'])
        self.assertEqual(len(adapter.species_calls), len(set(adapter.species_calls)))

    def test_partial_reaction_never_uploads_ts_or_kinetics(self):
        # The salvage path only touches species minima: no TS or kinetics
        # upload method exists on the stub, and the reaction bundle call is
        # marked partial (its caller strips ts_label + kinetics upstream).
        doc = _doc_with(ts_converged=False)
        adapter, _ = self._run(doc, _StubConfig(allow_partial_uploads=True))
        # Only reaction (partial) + species calls happened.
        self.assertTrue(all(is_partial for _, is_partial in adapter.reaction_calls))


if __name__ == '__main__':
    unittest.main()
