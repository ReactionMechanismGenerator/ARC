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
from arc.tckdb.sweep import _run_reaction_sweep, _run_ts_sweep


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


class _StubTSAdapter:
    """Records submit_computed_ts_from_output calls for the TS sweep."""

    def __init__(self, *, status='uploaded'):
        self.ts_calls = []  # list[(ts_label, reaction_label)]
        self._status = status

    def submit_computed_ts_from_output(self, *, output_doc, ts_record, reaction_record):
        self.ts_calls.append(
            (ts_record.get('label'), reaction_record.get('label'))
        )
        return _Outcome(status=self._status)


class _TSStubConfig:
    """The TS sweep reads no config fields today, but pass one for parity."""

    def __init__(self):
        self.upload_mode = 'computed_ts'


def _ts_doc(*, ts_converged, drop_reaction=False):
    """Reaction output doc with the TS ``converged`` flag set.

    ``drop_reaction`` removes the reactions list so the converged TS has
    no reaction referencing it (the no-reaction skip path).
    """
    doc = copy.deepcopy(_reaction_output_doc())
    for ts in doc['transition_states']:
        ts['converged'] = ts_converged
    if drop_reaction:
        doc['reactions'] = []
    return doc


class TestTSSweep(unittest.TestCase):
    def _run(self, doc):
        adapter = _StubTSAdapter()
        with redirect_stdout(io.StringIO()) as out:
            _run_ts_sweep(adapter=adapter, output_doc=doc, tckdb_config=_TSStubConfig())
        return adapter, out.getvalue()

    def test_converged_ts_uploaded_with_its_reaction(self):
        # A converged TS is uploaded once, paired with the reaction that
        # references it (so the embedded reactants/products resolve).
        doc = _ts_doc(ts_converged=True)
        adapter, _ = self._run(doc)
        self.assertEqual(
            adapter.ts_calls, [('TS0', 'CHO + CH4 <=> CH2O + CH3')],
        )

    def test_non_converged_ts_skipped(self):
        # Same eligibility gate as the species sweep: a non-converged TS
        # is never uploaded.
        doc = _ts_doc(ts_converged=False)
        adapter, out = self._run(doc)
        self.assertEqual(adapter.ts_calls, [])
        self.assertIn('0 converged TS', out)

    def test_converged_ts_without_reaction_skipped(self):
        # A converged TS with no reaction referencing it can't fill the
        # required embedded reaction; it is skipped (not uploaded), and
        # the reason is surfaced.
        doc = _ts_doc(ts_converged=True, drop_reaction=True)
        adapter, out = self._run(doc)
        self.assertEqual(adapter.ts_calls, [])
        self.assertIn('no reaction references the TS', out)

    def test_failed_upload_reported(self):
        doc = _ts_doc(ts_converged=True)
        adapter = _StubTSAdapter(status='failed')
        with redirect_stdout(io.StringIO()) as out:
            _run_ts_sweep(adapter=adapter, output_doc=doc, tckdb_config=_TSStubConfig())
        self.assertEqual(len(adapter.ts_calls), 1)
        self.assertIn('failed', out.getvalue())


if __name__ == '__main__':
    unittest.main()
