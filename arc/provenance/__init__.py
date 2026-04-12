"""
ARC provenance subpackage — directed acyclic graph for computational provenance.

Tracks the full chain of inputs, calculations, decisions, and outputs that
produce ARC's results.  Inspired by AiiDA's DAG model but adapted for ARC's
branching decision trees (TS guess evaluation, conformer selection,
troubleshooting loops).

Submodules:
    - ``nodes``: Node types, edge types, and their data classes.
    - ``graph``: ProvenanceGraph container with query and serialization.
"""

from arc.provenance.graph import ProvenanceGraph
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

__all__ = [
    'ProvenanceGraph',
    'ProvenanceNode',
    'CalculationNode',
    'DataNode',
    'DecisionNode',
    'ProvenanceEdge',
    'NodeType',
    'DataKind',
    'DecisionKind',
    'EdgeType',
]
