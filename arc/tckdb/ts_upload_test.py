#!/usr/bin/env python3
# encoding: utf-8

"""Tests for the standalone transition-state upload path
(``TCKDBAdapter.submit_computed_ts_from_output`` +
``_compose_transition_state_request``, POST ``/uploads/transition-states``).

The authoritative request schema (``TransitionStateUploadRequest``) lives
backend-side in ``app.schemas.workflows.transition_state_upload`` and is
not (yet) published in the standalone ``tckdb_schemas`` distribution the
way ``computed_species_upload`` / ``computed_reaction_upload`` are. So the
outer request wrapper is reconstructed here **on the real fragment
schemas** imported from ``tckdb_schemas`` — ``CalculationWithResultsPayload``,
``GeometryPayload``, and ``SpeciesEntryIdentityPayload`` — which carry all
the load-bearing validation (calc result/type consistency, XYZ text,
identity shape, ``tckdb_origin`` enum). The three thin outer classes below
mirror the backend source verbatim (verified against
``/uploads/transition-states``'s ``TransitionStateUploadRequest`` on the
TCKDB Pi); update them if the backend contract changes. No network POST is
performed — validation is purely against the pydantic schema.
"""

import copy
import unittest
from unittest import mock

from typing import Self

from pydantic import Field, model_validator
from tckdb_schemas.common import SchemaBase
from tckdb_schemas.enums import CalculationType
from tckdb_schemas.fragments.calculation import CalculationWithResultsPayload
from tckdb_schemas.fragments.geometry import GeometryPayload
from tckdb_schemas.fragments.identity import SpeciesEntryIdentityPayload
from tckdb_schemas.reaction_family import find_canonical_reaction_family

from arc.tckdb.adapter import (
    TCKDBAdapter,
    TRANSITION_STATE_ENDPOINT,
    TRANSITION_STATE_KIND,
)
from arc.tckdb.adapter_test import _aec_record, _mbac_record, _reaction_output_doc
from arc.tckdb.config import TCKDBArtifactConfig, TCKDBConfig


# ---------------------------------------------------------------------------
# Reconstructed outer request schema (mirrors the backend source; the
# fragments are the real ones from tckdb_schemas).
# ---------------------------------------------------------------------------


class TSReactionParticipantUpload(SchemaBase):
    species_entry: SpeciesEntryIdentityPayload
    note: str | None = None


class TSReactionUpload(SchemaBase):
    reversible: bool
    reaction_family: str | None = None
    reaction_family_source_note: str | None = None
    reactants: list[TSReactionParticipantUpload] = Field(min_length=1)
    products: list[TSReactionParticipantUpload] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reaction_family(self) -> Self:
        if self.reaction_family is None:
            if self.reaction_family_source_note is not None:
                raise ValueError("reaction_family_source_note requires reaction_family.")
            return self
        if find_canonical_reaction_family(self.reaction_family) is None:
            if self.reaction_family_source_note is None:
                raise ValueError(
                    "reaction_family_source_note is required when reaction_family "
                    "is not a supported canonical family."
                )
        return self


_ALLOWED_ADDITIONAL_TYPES = frozenset(
    {
        CalculationType.freq,
        CalculationType.sp,
        CalculationType.irc,
        CalculationType.path_search,
    }
)


class TransitionStateUploadRequest(SchemaBase):
    reaction: TSReactionUpload
    charge: int
    multiplicity: int = Field(ge=1)
    unmapped_smiles: str | None = None
    geometry: GeometryPayload
    primary_opt: CalculationWithResultsPayload
    additional_calculations: list[CalculationWithResultsPayload] = Field(
        default_factory=list
    )
    label: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def validate_primary_opt_is_opt(self) -> Self:
        if self.primary_opt.type != CalculationType.opt:
            raise ValueError(
                f"primary_opt must have type 'opt', got '{self.primary_opt.type.value}'."
            )
        return self

    @model_validator(mode="after")
    def validate_additional_calculation_types(self) -> Self:
        for calc in self.additional_calculations:
            if calc.type not in _ALLOWED_ADDITIONAL_TYPES:
                raise ValueError(
                    f"Additional calculation type '{calc.type.value}' is not allowed."
                )
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_ORIGIN_KINDS = frozenset({"executed", "reused_result", "imported", "derived"})


def _adapter(tmp_dir=None, *, upload=False):
    cfg = TCKDBConfig(
        enabled=True,
        base_url="http://localhost:8000/api/v1",
        payload_dir=tmp_dir or ".",
        api_key_env="X_TCKDB_API_KEY",
        project_label="proj-ts",
        upload_mode="computed_ts",
        upload=upload,
    )
    return TCKDBAdapter(cfg, project_directory=tmp_dir)


def _compose(*, with_irc=False):
    """Compose a standalone TS request from the shared reaction fixture."""
    doc = copy.deepcopy(_reaction_output_doc(with_irc=with_irc))
    doc["transition_states"][0]["converged"] = True
    ts = doc["transition_states"][0]
    rxn = doc["reactions"][0]
    payload = _adapter()._compose_transition_state_request(
        output_doc=doc, ts_record=ts, reaction_record=rxn,
    )
    return doc, ts, rxn, payload


class TestComposeTransitionStateRequest(unittest.TestCase):
    """(a) A converged TS composes a valid TransitionStateUploadRequest."""

    def test_payload_validates_against_schema(self):
        _, _, _, payload = _compose()
        # The load-bearing assertion: the full request (real fragments)
        # validates without a network POST.
        obj = TransitionStateUploadRequest(**payload)
        self.assertEqual(obj.primary_opt.type, CalculationType.opt)

    def test_primary_opt_required_and_is_opt(self):
        _, _, _, payload = _compose()
        self.assertIn("primary_opt", payload)
        self.assertEqual(payload["primary_opt"]["type"], "opt")
        # Wrapped result shape (computed-species style), not flattened.
        self.assertIn("opt_result", payload["primary_opt"])
        # No bundle-only cross-reference keys leak into the standalone calc.
        for banned in ("key", "depends_on", "geometry_key", "artifacts"):
            self.assertNotIn(banned, payload["primary_opt"])

    def test_additional_calculations_freq_sp(self):
        _, _, _, payload = _compose()
        types = [c["type"] for c in payload["additional_calculations"]]
        self.assertIn("freq", types)
        self.assertIn("sp", types)
        # The freq result carries the imaginary saddle mode.
        freq = next(c for c in payload["additional_calculations"] if c["type"] == "freq")
        self.assertEqual(freq["freq_result"]["n_imag"], 1)

    def test_additional_calculations_include_irc_when_present(self):
        _, _, _, payload = _compose(with_irc=True)
        types = [c["type"] for c in payload["additional_calculations"]]
        self.assertIn("irc", types)
        # Still validates with the IRC calc attached.
        TransitionStateUploadRequest(**payload)

    def test_reaction_embedded_from_reactants_products(self):
        _, _, _, payload = _compose()
        reaction = payload["reaction"]
        self.assertEqual(len(reaction["reactants"]), 2)
        self.assertEqual(len(reaction["products"]), 2)
        r_smiles = {p["species_entry"]["smiles"] for p in reaction["reactants"]}
        self.assertEqual(r_smiles, {"[CH]=O", "C"})
        p_smiles = {p["species_entry"]["smiles"] for p in reaction["products"]}
        self.assertEqual(p_smiles, {"C=O", "[CH3]"})
        # reversible is required by the schema; ARC defaults it to True.
        self.assertIs(reaction["reversible"], True)
        self.assertEqual(reaction["reaction_family"], "H_Abstraction")
        self.assertEqual(reaction["reaction_family_source_note"], "ARC-reported family")

    def test_geometry_and_charge_multiplicity(self):
        _, ts, _, payload = _compose()
        self.assertIn("xyz_text", payload["geometry"])
        # Normalized XYZ carries an atom-count header line.
        self.assertTrue(payload["geometry"]["xyz_text"].splitlines()[0].strip().isdigit())
        self.assertEqual(payload["charge"], ts["charge"])
        self.assertEqual(payload["multiplicity"], ts["multiplicity"])
        self.assertEqual(payload["label"], "TS0")
        # TS with no Lewis SMILES gets the reaction-SMILES traceability handle.
        self.assertEqual(payload["unmapped_smiles"], "[CH]=O.C>>C=O.[CH3]")

    def test_reaction_family_source_note_absent_without_family(self):
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        doc["reactions"][0].pop("family", None)
        payload = _adapter()._compose_transition_state_request(
            output_doc=doc,
            ts_record=doc["transition_states"][0],
            reaction_record=doc["reactions"][0],
        )
        self.assertNotIn("reaction_family", payload["reaction"])
        self.assertNotIn("reaction_family_source_note", payload["reaction"])
        TransitionStateUploadRequest(**payload)

    def test_missing_reactant_raises(self):
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        doc["reactions"][0]["reactant_labels"] = ["CHO", "GHOST"]
        with self.assertRaises(ValueError):
            _adapter()._compose_transition_state_request(
                output_doc=doc,
                ts_record=doc["transition_states"][0],
                reaction_record=doc["reactions"][0],
            )

    def test_whitespace_family_omitted(self):
        # A whitespace-only family must NOT be sent with a source note:
        # the backend normalizes it to None and would 422 on a source
        # note without a family. The producer omits both, mirroring the
        # backend's normalize_optional_text.
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        doc["reactions"][0]["family"] = "   "
        payload = _adapter()._compose_transition_state_request(
            output_doc=doc,
            ts_record=doc["transition_states"][0],
            reaction_record=doc["reactions"][0],
        )
        self.assertNotIn("reaction_family", payload["reaction"])
        self.assertNotIn("reaction_family_source_note", payload["reaction"])
        TransitionStateUploadRequest(**payload)

    def test_family_whitespace_padding_stripped(self):
        # A real family with surrounding whitespace is stripped (not
        # dropped) so it still resolves canonical server-side.
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        doc["reactions"][0]["family"] = "  H_Abstraction  "
        payload = _adapter()._compose_transition_state_request(
            output_doc=doc,
            ts_record=doc["transition_states"][0],
            reaction_record=doc["reactions"][0],
        )
        self.assertEqual(payload["reaction"]["reaction_family"], "H_Abstraction")
        TransitionStateUploadRequest(**payload)


class TestTSArtifactShortCircuit(unittest.TestCase):
    """Artifact upload isn't supported on the standalone TS endpoint.

    With ``config.artifacts.upload=True`` the composer must (a) NOT build
    /base64-encode any artifact (no wasted I/O, no build-then-drop), and
    (b) warn the user exactly once so the unsupported request isn't a
    silent no-op.
    """

    def _artifact_adapter(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=".",
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-ts",
            upload_mode="computed_ts",
            upload=False,
            artifacts=TCKDBArtifactConfig(upload=True, kinds=("output_log", "input")),
        )
        return TCKDBAdapter(cfg)

    def _doc(self):
        doc = copy.deepcopy(_reaction_output_doc(with_irc=True))
        doc["transition_states"][0]["converged"] = True
        # Give the TS log paths so _inline_artifacts_for_calc WOULD try to
        # build artifacts if it were ever called.
        doc["transition_states"][0]["opt_log"] = "/nonexistent/opt.log"
        doc["transition_states"][0]["freq_log"] = "/nonexistent/freq.log"
        return doc

    def test_inline_artifacts_never_called_and_no_artifacts_key(self):
        adapter = self._artifact_adapter()
        doc = self._doc()
        with mock.patch.object(
            adapter, "_inline_artifacts_for_calc",
        ) as inline:
            payload = adapter._compose_transition_state_request(
                output_doc=doc,
                ts_record=doc["transition_states"][0],
                reaction_record=doc["reactions"][0],
            )
        # Short-circuit: the artifact builder is never invoked for the
        # standalone TS path.
        inline.assert_not_called()
        for calc in [payload["primary_opt"], *payload.get("additional_calculations", [])]:
            self.assertNotIn("artifacts", calc)

    def test_warns_exactly_once_per_adapter(self):
        adapter = self._artifact_adapter()
        doc = self._doc()
        with mock.patch("arc.tckdb.adapter.logger") as log:
            for _ in range(3):
                adapter._compose_transition_state_request(
                    output_doc=doc,
                    ts_record=doc["transition_states"][0],
                    reaction_record=doc["reactions"][0],
                )
        artifact_warnings = [
            c for c in log.warning.call_args_list
            if "artifact upload" in str(c).lower()
            and "transition-state" in str(c).lower()
        ]
        self.assertEqual(len(artifact_warnings), 1)
        self.assertTrue(adapter._warned_ts_artifacts_unsupported)

    def test_no_warning_when_artifacts_disabled(self):
        # Default config (artifacts.upload=False) must not warn.
        adapter = _adapter()
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        with mock.patch("arc.tckdb.adapter.logger") as log:
            adapter._compose_transition_state_request(
                output_doc=doc,
                ts_record=doc["transition_states"][0],
                reaction_record=doc["reactions"][0],
            )
        artifact_warnings = [
            c for c in log.warning.call_args_list
            if "artifact upload" in str(c).lower()
        ]
        self.assertEqual(artifact_warnings, [])


class TestTSAppliedEnergyCorrections(unittest.TestCase):
    """AEC/BAC on the TS are dropped (no standalone slot) — deliberately."""

    def _doc_with_ts_aec(self):
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        doc["transition_states"][0]["applied_energy_corrections"] = [
            _aec_record(), _mbac_record(),
        ]
        return doc

    def test_aec_dropped_and_debug_logged(self):
        doc = self._doc_with_ts_aec()
        adapter = _adapter()
        with mock.patch("arc.tckdb.adapter.logger") as log:
            payload = adapter._compose_transition_state_request(
                output_doc=doc,
                ts_record=doc["transition_states"][0],
                reaction_record=doc["reactions"][0],
            )
        # No applied_energy_corrections anywhere in the standalone request.
        self.assertNotIn("applied_energy_corrections", payload)
        self.assertNotIn("applied_energy_corrections", payload["primary_opt"])
        # The drop is recorded at debug level (visible, intentional).
        aec_debug = [
            c for c in log.debug.call_args_list
            if "applied energy correction" in str(c).lower()
        ]
        self.assertEqual(len(aec_debug), 1)
        # Still a valid request without the corrections.
        TransitionStateUploadRequest(**payload)


class TestTSOriginKind(unittest.TestCase):
    """(d) origin_kind stays a valid enum on any calc carrying tckdb_origin."""

    def test_sp_reused_origin_kind_is_valid_enum(self):
        # The fixture declares only opt_level (no sp_level), so ARC reuses
        # the opt energy for SP -> the SP calc carries a reused_result
        # origin marker under parameters_json.tckdb_origin.
        _, _, _, payload = _compose()
        sp = next(c for c in payload["additional_calculations"] if c["type"] == "sp")
        origin = sp["parameters_json"]["tckdb_origin"]
        self.assertEqual(origin["origin_kind"], "reused_result")
        self.assertIn(origin["origin_kind"], _VALID_ORIGIN_KINDS)

    def test_all_origin_kinds_valid_and_schema_accepts_them(self):
        _, _, _, payload = _compose(with_irc=True)
        calcs = [payload["primary_opt"], *payload["additional_calculations"]]
        for calc in calcs:
            origin = (calc.get("parameters_json") or {}).get("tckdb_origin")
            if origin is not None:
                self.assertIn(origin["origin_kind"], _VALID_ORIGIN_KINDS)
        # The CalculationWithResultsPayload validator also runs
        # CalculationOriginMetadata.model_validate on any tckdb_origin, so
        # a full-request validation is an independent enum check.
        TransitionStateUploadRequest(**payload)


class TestTSIdempotency(unittest.TestCase):
    """(c) Content-hashed idempotency key: stable on replay, distinct on change."""

    def setUp(self):
        import tempfile
        import shutil
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-ts-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _submit(self, doc):
        adapter = _adapter(self.tmp, upload=False)
        return adapter.submit_computed_ts_from_output(
            output_doc=doc,
            ts_record=doc["transition_states"][0],
            reaction_record=doc["reactions"][0],
        )

    def test_same_content_replays_same_key(self):
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        first = self._submit(doc)
        second = self._submit(copy.deepcopy(doc))
        self.assertEqual(first.idempotency_key, second.idempotency_key)
        self.assertTrue(first.idempotency_key.startswith("arc:proj-ts:TS0:"))

    def test_changed_geometry_changes_key(self):
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        first = self._submit(doc)
        mutated = copy.deepcopy(doc)
        mutated["transition_states"][0]["xyz"] = (
            "C 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.9 0.9 0.0"
        )
        second = self._submit(mutated)
        self.assertNotEqual(first.idempotency_key, second.idempotency_key)


class TestTSUploadWiring(unittest.TestCase):
    """The adapter routes the TS payload through the shared submit path."""

    def test_disabled_adapter_returns_none(self):
        cfg = TCKDBConfig(enabled=False)
        adapter = TCKDBAdapter(cfg)
        doc = _reaction_output_doc()
        self.assertIsNone(
            adapter.submit_computed_ts_from_output(
                output_doc=doc,
                ts_record=doc["transition_states"][0],
                reaction_record=doc["reactions"][0],
            )
        )

    def test_offline_writes_transition_state_payload(self):
        import tempfile
        import shutil
        import json
        tmp = tempfile.mkdtemp(prefix="arc-tckdb-ts-wire-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        doc = copy.deepcopy(_reaction_output_doc())
        doc["transition_states"][0]["converged"] = True
        adapter = _adapter(tmp, upload=False)
        outcome = adapter.submit_computed_ts_from_output(
            output_doc=doc,
            ts_record=doc["transition_states"][0],
            reaction_record=doc["reactions"][0],
        )
        self.assertEqual(outcome.status, "skipped")  # upload=False
        # Payload lands under the transition_state subdir.
        self.assertIn("transition_state", str(outcome.payload_path))
        written = json.loads(outcome.payload_path.read_text())
        TransitionStateUploadRequest(**written)
        # Sidecar records the standalone endpoint + kind.
        sidecar = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sidecar["endpoint"], TRANSITION_STATE_ENDPOINT)
        self.assertEqual(sidecar["payload_kind"], TRANSITION_STATE_KIND)


if __name__ == "__main__":
    unittest.main()
