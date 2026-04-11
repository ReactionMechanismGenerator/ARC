"""Tests for the arc.provenance package — nodes, edges, and ProvenanceGraph."""

import os
import shutil
import tempfile
import unittest

from arc.provenance.graph import SCHEMA_VERSION, ProvenanceGraph
from arc.provenance.nodes import (
    CalculationNode,
    DataKind,
    DataNode,
    DecisionKind,
    DecisionNode,
    EdgeType,
    NodeType,
    ProvenanceEdge,
    ProvenanceNode,
)


class TestEnums(unittest.TestCase):
    """Verify that enums are str-based and contain expected values."""

    def test_node_type_is_str(self):
        self.assertIsInstance(NodeType.species, str)
        self.assertEqual(NodeType.calculation, 'calculation')

    def test_data_kind_values(self):
        self.assertIn('geometry', [dk.value for dk in DataKind])
        self.assertIn('energy', [dk.value for dk in DataKind])

    def test_decision_kind_values(self):
        expected = {'conformer_selection', 'ts_guess_clustering', 'ts_guess_selection',
                    'ts_guess_selection_failed', 'ts_validation_freq', 'ts_validation_nmd',
                    'ts_validation_irc', 'ts_switch', 'job_troubleshooting', 'ts_method_spawning'}
        actual = {dk.value for dk in DecisionKind}
        self.assertEqual(expected, actual)

    def test_edge_type_values(self):
        self.assertIn('input_of', [et.value for et in EdgeType])
        self.assertIn('selected_by', [et.value for et in EdgeType])
        self.assertIn('rejected_by', [et.value for et in EdgeType])


class TestProvenanceNode(unittest.TestCase):
    """Test the base ProvenanceNode class."""

    def test_creation(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species, label='ethanol')
        self.assertEqual(node.node_id, 'species_1')
        self.assertEqual(node.node_type, 'species')
        self.assertEqual(node.label, 'ethanol')
        self.assertIsNotNone(node.timestamp)

    def test_as_dict_sparse(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species)
        d = node.as_dict()
        self.assertIn('node_id', d)
        self.assertIn('node_type', d)
        self.assertNotIn('label', d)
        self.assertNotIn('metadata', d)

    def test_as_dict_with_metadata(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species,
                              label='H2O', metadata={'is_ts': True})
        d = node.as_dict()
        self.assertEqual(d['metadata'], {'is_ts': True})

    def test_from_dict_roundtrip(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species,
                              label='ethanol', metadata={'is_ts': False})
        d = node.as_dict()
        restored = ProvenanceNode.from_dict(d)
        self.assertEqual(restored.node_id, 'species_1')
        self.assertEqual(restored.node_type, 'species')
        self.assertEqual(restored.label, 'ethanol')

    def test_from_dict_dispatches_to_subclass(self):
        d = {'node_id': 'calc_1', 'node_type': 'calculation', 'job_name': 'opt_a1'}
        restored = ProvenanceNode.from_dict(d)
        self.assertIsInstance(restored, CalculationNode)
        self.assertEqual(restored.job_name, 'opt_a1')

    def test_repr(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species, label='ethanol')
        self.assertIn('species_1', repr(node))


class TestCalculationNode(unittest.TestCase):
    """Test CalculationNode creation and serialization."""

    def test_creation(self):
        node = CalculationNode(node_id='calc_1', label='ethanol', job_name='opt_a1',
                               job_type='opt', job_adapter='gaussian',
                               level='wb97xd/def2-tzvp', status='done')
        self.assertEqual(node.node_type, 'calculation')
        self.assertEqual(node.job_name, 'opt_a1')
        self.assertEqual(node.status, 'done')

    def test_as_dict_sparse(self):
        node = CalculationNode(node_id='calc_1', label='ethanol', job_name='opt_a1')
        d = node.as_dict()
        self.assertIn('job_name', d)
        self.assertNotIn('job_adapter', d)
        self.assertNotIn('ess_trsh_methods', d)

    def test_from_dict_roundtrip(self):
        node = CalculationNode(node_id='calc_1', label='ethanol', job_name='opt_a1',
                               job_type='opt', status='errored',
                               ess_trsh_methods=['SCF=QC', 'int=grid=ultrafine'])
        d = node.as_dict()
        restored = CalculationNode.from_dict(d)
        self.assertEqual(restored.job_name, 'opt_a1')
        self.assertEqual(restored.status, 'errored')
        self.assertEqual(restored.ess_trsh_methods, ['SCF=QC', 'int=grid=ultrafine'])
        self.assertIsNone(restored.conformer)


class TestDataNode(unittest.TestCase):
    """Test DataNode creation and serialization."""

    def test_creation(self):
        node = DataNode(node_id='data_1', label='ethanol',
                        data_kind=DataKind.energy, value=-79.123456)
        self.assertEqual(node.node_type, 'data')
        self.assertEqual(node.data_kind, 'energy')
        self.assertEqual(node.value, -79.123456)

    def test_from_dict_roundtrip(self):
        node = DataNode(node_id='data_1', label='ethanol',
                        data_kind=DataKind.frequencies, value=[3200.5, 1500.3, 800.1])
        d = node.as_dict()
        restored = DataNode.from_dict(d)
        self.assertEqual(restored.data_kind, 'frequencies')
        self.assertEqual(restored.value, [3200.5, 1500.3, 800.1])


class TestDecisionNode(unittest.TestCase):
    """Test DecisionNode creation and serialization."""

    def test_creation(self):
        node = DecisionNode(node_id='decision_1', label='TS0',
                            decision_kind=DecisionKind.ts_guess_selection,
                            outcome='Selected TSGuess #3 (energy=-150.2 kJ/mol)')
        self.assertEqual(node.node_type, 'decision')
        self.assertEqual(node.decision_kind, 'ts_guess_selection')
        self.assertIn('TSGuess #3', node.outcome)

    def test_from_dict_roundtrip(self):
        node = DecisionNode(node_id='decision_1', label='TS0',
                            decision_kind=DecisionKind.job_troubleshooting,
                            criteria={'error_keywords': ['SCF', 'Memory'],
                                      'applied': 'SCF=QC'},
                            outcome='Retrying with SCF=QC')
        d = node.as_dict()
        restored = DecisionNode.from_dict(d)
        self.assertEqual(restored.decision_kind, 'job_troubleshooting')
        self.assertEqual(restored.criteria['error_keywords'], ['SCF', 'Memory'])
        self.assertEqual(restored.outcome, 'Retrying with SCF=QC')


class TestProvenanceEdge(unittest.TestCase):
    """Test ProvenanceEdge creation and serialization."""

    def test_creation(self):
        edge = ProvenanceEdge(source_id='species_1', target_id='calc_1',
                              edge_type=EdgeType.input_of)
        self.assertEqual(edge.source_id, 'species_1')
        self.assertEqual(edge.edge_type, 'input_of')

    def test_as_dict_sparse(self):
        edge = ProvenanceEdge(source_id='a', target_id='b', edge_type=EdgeType.output_of)
        d = edge.as_dict()
        self.assertNotIn('metadata', d)

    def test_from_dict_roundtrip(self):
        edge = ProvenanceEdge(source_id='calc_1', target_id='data_1',
                              edge_type=EdgeType.output_of,
                              metadata={'reason': 'rerun'})
        d = edge.as_dict()
        restored = ProvenanceEdge.from_dict(d)
        self.assertEqual(restored.source_id, 'calc_1')
        self.assertEqual(restored.metadata, {'reason': 'rerun'})

    def test_repr(self):
        edge = ProvenanceEdge(source_id='a', target_id='b', edge_type=EdgeType.selected_by)
        self.assertIn('selected_by', repr(edge))


class TestProvenanceGraph(unittest.TestCase):
    """Test ProvenanceGraph CRUD, traversal, query, and serialization."""

    def setUp(self):
        self.graph = ProvenanceGraph(project='test_project')

    def test_add_species_node(self):
        nid = self.graph.add_species_node(label='ethanol')
        self.assertIn(nid, self.graph.nodes)
        self.assertEqual(self.graph.nodes[nid].node_type, 'species')
        self.assertEqual(self.graph.nodes[nid].label, 'ethanol')

    def test_add_calculation_node(self):
        nid = self.graph.add_calculation_node(label='ethanol', job_name='opt_a1',
                                              job_type='opt', status='pending')
        node = self.graph.get_node(nid)
        self.assertIsInstance(node, CalculationNode)
        self.assertEqual(node.job_name, 'opt_a1')

    def test_add_data_node(self):
        nid = self.graph.add_data_node(label='ethanol', data_kind=DataKind.energy,
                                       value=-79.5)
        node = self.graph.get_node(nid)
        self.assertIsInstance(node, DataNode)
        self.assertEqual(node.value, -79.5)

    def test_add_decision_node(self):
        nid = self.graph.add_decision_node(label='TS0',
                                           decision_kind=DecisionKind.ts_guess_selection,
                                           outcome='Selected TSG #2')
        node = self.graph.get_node(nid)
        self.assertIsInstance(node, DecisionNode)
        self.assertEqual(node.outcome, 'Selected TSG #2')

    def test_node_id_auto_increment(self):
        id1 = self.graph.add_species_node(label='A')
        id2 = self.graph.add_species_node(label='B')
        id3 = self.graph.add_calculation_node(label='A', job_name='opt_a1')
        self.assertEqual(id1, 'species_1')
        self.assertEqual(id2, 'species_2')
        self.assertEqual(id3, 'calc_3')

    def test_duplicate_node_skipped(self):
        node = ProvenanceNode(node_id='species_1', node_type=NodeType.species, label='X')
        self.graph.add_node(node)
        self.graph.add_node(node)
        self.assertEqual(len(self.graph.nodes), 1)

    def test_add_edge(self):
        sid = self.graph.add_species_node(label='ethanol')
        cid = self.graph.add_calculation_node(label='ethanol', job_name='opt_a1')
        edge = self.graph.add_edge(sid, cid, EdgeType.input_of)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(edge.edge_type, 'input_of')

    def test_get_edges_from_and_to(self):
        sid = self.graph.add_species_node(label='A')
        c1 = self.graph.add_calculation_node(label='A', job_name='opt_a1')
        c2 = self.graph.add_calculation_node(label='A', job_name='freq_a2')
        self.graph.add_edge(sid, c1, EdgeType.input_of)
        self.graph.add_edge(sid, c2, EdgeType.input_of)
        self.assertEqual(len(self.graph.get_edges_from(sid)), 2)
        self.assertEqual(len(self.graph.get_edges_to(c1)), 1)

    def test_get_nodes_by_type(self):
        self.graph.add_species_node(label='A')
        self.graph.add_species_node(label='B')
        self.graph.add_calculation_node(label='A', job_name='opt')
        species_nodes = self.graph.get_nodes_by_type(NodeType.species)
        self.assertEqual(len(species_nodes), 2)
        calc_nodes = self.graph.get_nodes_by_type(NodeType.calculation)
        self.assertEqual(len(calc_nodes), 1)

    def test_get_nodes_by_type_with_label_filter(self):
        self.graph.add_species_node(label='A')
        self.graph.add_species_node(label='B')
        self.graph.add_calculation_node(label='A', job_name='opt')
        self.graph.add_calculation_node(label='B', job_name='opt')
        a_calcs = self.graph.get_nodes_by_type(NodeType.calculation, label='A')
        self.assertEqual(len(a_calcs), 1)

    def test_get_nodes_by_label(self):
        self.graph.add_species_node(label='ethanol')
        self.graph.add_calculation_node(label='ethanol', job_name='opt')
        self.graph.add_calculation_node(label='methane', job_name='opt')
        eth_nodes = self.graph.get_nodes_by_label('ethanol')
        self.assertEqual(len(eth_nodes), 2)

    def test_find_species_node(self):
        sid = self.graph.add_species_node(label='ethanol')
        self.assertEqual(self.graph.find_species_node('ethanol'), sid)
        self.assertIsNone(self.graph.find_species_node('missing'))

    def test_find_calc_node(self):
        self.graph.add_calculation_node(label='A', job_name='opt_a1')
        cid = self.graph.find_calc_node('A', 'opt_a1')
        self.assertIsNotNone(cid)
        self.assertIsNone(self.graph.find_calc_node('A', 'missing'))

    def test_update_node(self):
        cid = self.graph.add_calculation_node(label='A', job_name='opt', status='pending')
        self.assertTrue(self.graph.update_node(cid, status='done', run_time='00:05:30'))
        node = self.graph.get_node(cid)
        self.assertEqual(node.status, 'done')
        self.assertEqual(node.run_time, '00:05:30')

    def test_update_node_missing(self):
        self.assertFalse(self.graph.update_node('nonexistent', status='done'))

    def test_get_edges_by_type(self):
        sid = self.graph.add_species_node(label='A')
        c1 = self.graph.add_calculation_node(label='A', job_name='opt')
        d1 = self.graph.add_data_node(label='A', data_kind=DataKind.energy)
        self.graph.add_edge(sid, c1, EdgeType.input_of)
        self.graph.add_edge(c1, d1, EdgeType.output_of)
        self.assertEqual(len(self.graph.get_edges_by_type(EdgeType.input_of)), 1)
        self.assertEqual(len(self.graph.get_edges_by_type(EdgeType.output_of)), 1)
        self.assertEqual(len(self.graph.get_edges_by_type(EdgeType.selected_by)), 0)

    # ── Traversal ────────────────────────────────────────────────────────────

    def test_descendants(self):
        """species -> calc -> data -> decision"""
        sid = self.graph.add_species_node(label='A')
        cid = self.graph.add_calculation_node(label='A', job_name='opt')
        did = self.graph.add_data_node(label='A', data_kind=DataKind.geometry)
        dec = self.graph.add_decision_node(label='A', decision_kind=DecisionKind.conformer_selection)
        self.graph.add_edge(sid, cid, EdgeType.input_of)
        self.graph.add_edge(cid, did, EdgeType.output_of)
        self.graph.add_edge(did, dec, EdgeType.selected_by)
        desc = self.graph.descendants(sid)
        self.assertEqual(set(desc), {cid, did, dec})
        self.assertNotIn(sid, desc)

    def test_ancestors(self):
        """Reverse traversal."""
        sid = self.graph.add_species_node(label='A')
        cid = self.graph.add_calculation_node(label='A', job_name='opt')
        did = self.graph.add_data_node(label='A', data_kind=DataKind.energy)
        self.graph.add_edge(sid, cid, EdgeType.input_of)
        self.graph.add_edge(cid, did, EdgeType.output_of)
        anc = self.graph.ancestors(did)
        self.assertEqual(set(anc), {sid, cid})

    def test_no_descendants(self):
        sid = self.graph.add_species_node(label='A')
        self.assertEqual(self.graph.descendants(sid), [])

    # ── Query ────────────────────────────────────────────────────────────────

    def test_query_by_node_type(self):
        self.graph.add_species_node(label='A')
        self.graph.add_calculation_node(label='A', job_name='opt')
        results = self.graph.query(node_type=NodeType.species)
        self.assertEqual(len(results), 1)

    def test_query_by_decision_kind(self):
        self.graph.add_decision_node(label='A', decision_kind=DecisionKind.ts_guess_selection)
        self.graph.add_decision_node(label='A', decision_kind=DecisionKind.job_troubleshooting)
        results = self.graph.query(decision_kind=DecisionKind.ts_guess_selection)
        self.assertEqual(len(results), 1)

    def test_query_by_status(self):
        self.graph.add_calculation_node(label='A', job_name='opt', status='done')
        self.graph.add_calculation_node(label='A', job_name='freq', status='errored')
        done = self.graph.query(status='done')
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].job_name, 'opt')

    def test_query_combined_filters(self):
        self.graph.add_calculation_node(label='A', job_name='opt', status='done')
        self.graph.add_calculation_node(label='B', job_name='opt', status='done')
        results = self.graph.query(node_type=NodeType.calculation, label='A', status='done')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, 'A')

    # ── Serialization ────────────────────────────────────────────────────────

    def test_as_dict_structure(self):
        self.graph.add_species_node(label='A')
        d = self.graph.as_dict()
        self.assertEqual(d['schema_version'], SCHEMA_VERSION)
        self.assertEqual(d['project'], 'test_project')
        self.assertIsInstance(d['nodes'], list)
        self.assertIsInstance(d['edges'], list)

    def test_from_dict_roundtrip(self):
        sid = self.graph.add_species_node(label='ethanol')
        cid = self.graph.add_calculation_node(label='ethanol', job_name='opt_a1',
                                              status='done')
        self.graph.add_edge(sid, cid, EdgeType.input_of)
        d = self.graph.as_dict()
        restored = ProvenanceGraph.from_dict(d)
        self.assertEqual(len(restored.nodes), 2)
        self.assertEqual(len(restored.edges), 1)
        self.assertEqual(restored.project, 'test_project')
        self.assertIsInstance(restored.get_node(cid), CalculationNode)
        self.assertEqual(restored.get_node(cid).status, 'done')

    def test_restart_continues_counter(self):
        """After loading a graph, new node IDs should not collide with existing ones."""
        self.graph.add_species_node(label='A')
        self.graph.add_species_node(label='B')
        self.graph.add_calculation_node(label='A', job_name='opt')
        d = self.graph.as_dict()
        restored = ProvenanceGraph.from_dict(d)
        new_id = restored.add_species_node(label='C')
        # _counter should be at least 3 (from species_1, species_2, calc_3),
        # so next ID should be species_4 or higher
        self.assertNotIn(new_id, ['species_1', 'species_2', 'calc_3'])

    def test_save_and_load(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        path = os.path.join(tmp_dir, 'provenance_graph.yml')
        sid = self.graph.add_species_node(label='ethanol')
        cid = self.graph.add_calculation_node(label='ethanol', job_name='opt_a1',
                                              job_type='opt', status='done')
        did = self.graph.add_data_node(label='ethanol', data_kind=DataKind.energy,
                                       value=-79.5)
        dec = self.graph.add_decision_node(label='ethanol',
                                           decision_kind=DecisionKind.conformer_selection,
                                           outcome='Selected conformer #0')
        self.graph.add_edge(sid, cid, EdgeType.input_of)
        self.graph.add_edge(cid, did, EdgeType.output_of)
        self.graph.add_edge(did, dec, EdgeType.selected_by)
        self.graph.save(path)
        self.assertTrue(os.path.isfile(path))
        loaded = ProvenanceGraph.load(path)
        self.assertEqual(len(loaded.nodes), 4)
        self.assertEqual(len(loaded.edges), 3)
        self.assertIsInstance(loaded.get_node(cid), CalculationNode)
        self.assertIsInstance(loaded.get_node(did), DataNode)
        self.assertIsInstance(loaded.get_node(dec), DecisionNode)

    def test_len_and_repr(self):
        self.assertEqual(len(self.graph), 0)
        self.graph.add_species_node(label='A')
        self.assertEqual(len(self.graph), 1)
        self.assertIn('test_project', repr(self.graph))


class TestProvenanceGraphWorkflow(unittest.TestCase):
    """
    Integration-style test: build a realistic provenance graph for a species
    going through opt → freq → sp, with a troubleshooting retry on freq.
    """

    def test_realistic_workflow(self):
        g = ProvenanceGraph(project='workflow_test')

        # Species initialized
        sid = g.add_species_node(label='ethanol')

        # Opt job succeeds
        opt_id = g.add_calculation_node(label='ethanol', job_name='opt_a1',
                                        job_type='opt', status='done')
        g.add_edge(sid, opt_id, EdgeType.input_of)
        opt_geo = g.add_data_node(label='ethanol', data_kind=DataKind.geometry,
                                  source_path='calcs/opt_a1/output.log')
        g.add_edge(opt_id, opt_geo, EdgeType.output_of)

        # Freq job fails
        freq1_id = g.add_calculation_node(label='ethanol', job_name='freq_a2',
                                          job_type='freq', status='errored')
        g.add_edge(opt_geo, freq1_id, EdgeType.input_of)

        # Troubleshooting decision
        trsh_id = g.add_decision_node(label='ethanol',
                                      decision_kind=DecisionKind.job_troubleshooting,
                                      criteria={'error_keywords': ['SCF']},
                                      outcome='Retrying with SCF=QC')
        g.add_edge(freq1_id, trsh_id, EdgeType.troubleshot_by)

        # Freq job retried and succeeds
        freq2_id = g.add_calculation_node(label='ethanol', job_name='freq_a3',
                                          job_type='freq', status='done',
                                          ess_trsh_methods=['SCF=QC'])
        g.add_edge(trsh_id, freq2_id, EdgeType.spawned_by)
        g.add_edge(freq1_id, freq2_id, EdgeType.retried_as)
        freq_data = g.add_data_node(label='ethanol', data_kind=DataKind.frequencies,
                                    value=[3200.5, 1500.3])
        g.add_edge(freq2_id, freq_data, EdgeType.output_of)

        # SP job succeeds
        sp_id = g.add_calculation_node(label='ethanol', job_name='sp_a4',
                                       job_type='sp', status='done')
        g.add_edge(opt_geo, sp_id, EdgeType.input_of)
        sp_energy = g.add_data_node(label='ethanol', data_kind=DataKind.energy,
                                    value=-79.123456)
        g.add_edge(sp_id, sp_energy, EdgeType.output_of)

        # Verify graph structure
        self.assertEqual(len(g.nodes), 9)
        self.assertEqual(len(g.edges), 9)

        # Verify traversal: ancestors of the final energy should trace back to species
        anc = g.ancestors(sp_energy)
        self.assertIn(sid, anc)
        self.assertIn(opt_id, anc)
        self.assertIn(sp_id, anc)

        # Verify query: find all troubleshooting decisions
        trsh_decisions = g.query(decision_kind=DecisionKind.job_troubleshooting)
        self.assertEqual(len(trsh_decisions), 1)
        self.assertEqual(trsh_decisions[0].criteria['error_keywords'], ['SCF'])

        # Verify query: find all errored calculations
        errored = g.query(node_type=NodeType.calculation, status='errored')
        self.assertEqual(len(errored), 1)
        self.assertEqual(errored[0].job_name, 'freq_a2')

        # Verify traversal: descendants of the troubleshooting decision
        # should include the retried freq job and its output
        desc = g.descendants(trsh_id)
        self.assertIn(freq2_id, desc)
        self.assertIn(freq_data, desc)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases identified during code review."""

    def setUp(self):
        self.graph = ProvenanceGraph(project='edge_case_test')

    def test_add_edge_warns_on_nonexistent_nodes(self):
        """add_edge should still work but log warnings for missing nodes."""
        sid = self.graph.add_species_node(label='A')
        edge = self.graph.add_edge(sid, 'nonexistent_target', EdgeType.input_of)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(edge.target_id, 'nonexistent_target')

    def test_roundtrip_preserves_zero_value(self):
        """DataNode with value=0 (falsy) must survive serialization."""
        nid = self.graph.add_data_node(label='A', data_kind=DataKind.energy, value=0)
        d = self.graph.as_dict()
        restored = ProvenanceGraph.from_dict(d)
        node = restored.get_node(nid)
        self.assertIsInstance(node, DataNode)
        self.assertEqual(node.value, 0)

    def test_roundtrip_preserves_false_in_metadata(self):
        """Metadata with False values must survive serialization."""
        node = ProvenanceNode(node_id='species_99', node_type=NodeType.species,
                              label='X', metadata={'is_ts': False, 'converged': False})
        self.graph.add_node(node)
        d = self.graph.as_dict()
        restored = ProvenanceGraph.from_dict(d)
        restored_node = restored.get_node('species_99')
        self.assertEqual(restored_node.metadata['is_ts'], False)
        self.assertEqual(restored_node.metadata['converged'], False)

    def test_roundtrip_omits_empty_ess_trsh_methods(self):
        """CalculationNode with ess_trsh_methods=[] should omit it from dict."""
        node = CalculationNode(node_id='calc_99', label='A', ess_trsh_methods=[])
        d = node.as_dict()
        self.assertNotIn('ess_trsh_methods', d)

    def test_ancestors_with_diamond_dependency(self):
        """DAG diamond: A -> B -> D, A -> C -> D — ancestors(D) = {A, B, C}."""
        a = self.graph.add_species_node(label='A')
        b = self.graph.add_calculation_node(label='A', job_name='opt')
        c = self.graph.add_calculation_node(label='A', job_name='freq')
        d = self.graph.add_data_node(label='A', data_kind=DataKind.energy)
        self.graph.add_edge(a, b, EdgeType.input_of)
        self.graph.add_edge(a, c, EdgeType.input_of)
        self.graph.add_edge(b, d, EdgeType.output_of)
        self.graph.add_edge(c, d, EdgeType.output_of)
        anc = self.graph.ancestors(d)
        self.assertEqual(set(anc), {a, b, c})

    def test_descendants_handles_self_loop(self):
        """If a self-loop is accidentally created, traversal should not infinite-loop."""
        nid = self.graph.add_species_node(label='A')
        self.graph.add_edge(nid, nid, EdgeType.input_of)
        desc = self.graph.descendants(nid)
        self.assertIn(nid, desc)

    def test_query_enum_and_string_equivalence(self):
        """Query with NodeType enum and plain string should return identical results."""
        self.graph.add_calculation_node(label='A', job_name='opt', status='done')
        r1 = self.graph.query(node_type=NodeType.calculation)
        r2 = self.graph.query(node_type='calculation')
        self.assertEqual(len(r1), len(r2))
        self.assertEqual(r1[0].node_id, r2[0].node_id)

    def test_counter_with_mixed_prefixes_after_restart(self):
        """Counter should track max across ALL prefixes, not per-prefix."""
        self.graph.add_species_node(label='A')     # species_1
        self.graph.add_species_node(label='B')     # species_2
        self.graph.add_calculation_node(label='A', job_name='opt')  # calc_3
        d = self.graph.as_dict()
        restored = ProvenanceGraph.from_dict(d)
        # Counter should be >= 3, so next ID suffix is >= 4
        new_id = restored.add_data_node(label='A', data_kind=DataKind.energy)
        suffix = int(new_id.split('_')[-1])
        self.assertGreaterEqual(suffix, 4)

    def test_render_all_edge_types(self):
        """Verify render_provenance_graph handles every EdgeType without errors."""
        try:
            import graphviz as gv_mod
        except ImportError:
            self.skipTest('graphviz not installed')
        from arc.plotter import render_provenance_graph
        g = ProvenanceGraph(project='edge_type_test')
        n1 = g.add_species_node(label='A')
        n2 = g.add_calculation_node(label='A', job_name='opt', status='done')
        g.add_data_node(label='A', data_kind=DataKind.energy)
        g.add_decision_node(label='A', decision_kind=DecisionKind.conformer_selection)
        g.add_calculation_node(label='A', job_name='opt2', status='errored')
        for et in list(EdgeType):
            g.add_edge(n1, n2, et)
        gv = render_provenance_graph(g, run_label='test')
        dot = gv.source
        self.assertIn('species_1', dot)
        self.assertIn('calc_2', dot)

    def test_render_none_labels(self):
        """Nodes with label=None should render using node_id as fallback."""
        try:
            import graphviz as gv_mod
        except ImportError:
            self.skipTest('graphviz not installed')
        from arc.plotter import render_provenance_graph
        g = ProvenanceGraph(project='none_label_test')
        g.add_species_node(label=None)
        g.add_calculation_node(label=None, job_name='opt', status='pending')
        gv = render_provenance_graph(g, run_label='test')
        dot = gv.source
        # Should not crash; node_id is used as fallback for species
        self.assertIn('species_1', dot)


if __name__ == '__main__':
    unittest.main()
