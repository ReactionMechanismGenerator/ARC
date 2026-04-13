"""
Provenance node and edge types for the ARC provenance DAG.

Defines the fundamental building blocks of the provenance graph:

- **Enums**: ``NodeType``, ``DataKind``, ``DecisionKind``, ``EdgeType``
  classify nodes and edges.
- **Node classes**: ``ProvenanceNode`` (base), ``CalculationNode``,
  ``DataNode``, ``DecisionNode`` represent vertices in the DAG.
- **Edge class**: ``ProvenanceEdge`` represents a directed, typed
  relationship between two nodes.

All classes follow the ``as_dict()`` / ``from_dict()`` serialization
pattern used throughout ARC (see ``arc.job.pipe.pipe_state``).
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


def _enum_val(val):
    """Extract the plain string value from a str-Enum or pass through a string."""
    return val.value if isinstance(val, Enum) else val


# ── Enums ────────────────────────────────────────────────────────────────────


class NodeType(str, Enum):
    """Types of nodes in the provenance DAG."""
    species = 'species'
    data = 'data'
    calculation = 'calculation'
    decision = 'decision'


class DataKind(str, Enum):
    """Sub-classification for DataNode content."""
    geometry = 'geometry'
    energy = 'energy'
    frequencies = 'frequencies'
    imaginary_freq = 'imaginary_freq'
    irc_path = 'irc_path'
    conformer_set = 'conformer_set'
    ts_guess_set = 'ts_guess_set'
    thermo = 'thermo'
    kinetics = 'kinetics'


class DecisionKind(str, Enum):
    """Sub-classification for DecisionNode decisions."""
    conformer_selection = 'conformer_selection'
    ts_guess_clustering = 'ts_guess_clustering'
    ts_guess_selection = 'ts_guess_selection'
    ts_guess_selection_failed = 'ts_guess_selection_failed'
    ts_validation_freq = 'ts_validation_freq'
    ts_validation_nmd = 'ts_validation_nmd'
    ts_validation_irc = 'ts_validation_irc'
    ts_validation_e0 = 'ts_validation_e0'
    ts_validation_e_elect = 'ts_validation_e_elect'
    ts_switch = 'ts_switch'
    job_troubleshooting = 'job_troubleshooting'
    ts_method_spawning = 'ts_method_spawning'
    convergence_confirmed = 'convergence_confirmed'


class EdgeType(str, Enum):
    """Types of directed edges in the provenance DAG."""
    input_of = 'input_of'
    output_of = 'output_of'
    triggered_by = 'triggered_by'
    selected_by = 'selected_by'
    rejected_by = 'rejected_by'
    spawned_by = 'spawned_by'
    troubleshot_by = 'troubleshot_by'
    belongs_to = 'belongs_to'
    retried_as = 'retried_as'
    fine_of = 'fine_of'


# ── Node classes ─────────────────────────────────────────────────────────────


class ProvenanceNode(object):
    """
    Base class for a node in the provenance DAG.

    Args:
        node_id (str): Unique identifier (e.g. ``'species_0'``, ``'calc_17'``).
        node_type (str): One of :class:`NodeType` values.
        label (str, optional): Species label this node is associated with.
        timestamp (str, optional): ISO 8601 creation timestamp.
            Auto-generated if not provided.
        metadata (dict, optional): Arbitrary extra key-value data.
    """

    def __init__(self,
                 node_id: str,
                 node_type: str,
                 label: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 ):
        self.node_id = node_id
        self.node_type = _enum_val(node_type)
        self.label = label
        self.timestamp = timestamp or datetime.datetime.now().isoformat(timespec='seconds')
        self.metadata = metadata

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to a sparse dict (None and empty values omitted)."""
        d: Dict[str, Any] = {
            'node_id': self.node_id,
            'node_type': self.node_type,
        }
        if self.label is not None:
            d['label'] = self.label
        if self.timestamp is not None:
            d['timestamp'] = self.timestamp
        if self.metadata:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProvenanceNode':
        """Reconstruct a ProvenanceNode (or appropriate subclass) from a dict."""
        node_type = d.get('node_type', '')
        # Dispatch to the correct subclass based on node_type.
        # Keys use plain strings so YAML-deserialized values match.
        subclass_map = {
            'calculation': CalculationNode,
            'data': DataNode,
            'decision': DecisionNode,
        }
        target_cls = subclass_map.get(node_type, cls)
        if target_cls is not cls:
            return target_cls.from_dict(d)
        obj = object.__new__(cls)
        obj.node_id = d['node_id']
        obj.node_type = d.get('node_type', '')
        obj.label = d.get('label')
        obj.timestamp = d.get('timestamp')
        obj.metadata = d.get('metadata')
        return obj

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.node_id!r}, type={self.node_type!r}, label={self.label!r})'


class CalculationNode(ProvenanceNode):
    """
    A computational job node (opt, freq, sp, scan, tsg, irc, composite, etc.).

    Args:
        node_id (str): Unique identifier.
        label (str, optional): Species label.
        job_name (str, optional): ARC job name (e.g. ``'opt_a1'``).
        job_type (str, optional): Job type (e.g. ``'opt'``, ``'freq'``).
        job_adapter (str, optional): ESS adapter (e.g. ``'gaussian'``).
        level (str, optional): Level of theory string.
        status (str, optional): Job outcome: ``'pending'``, ``'done'``, ``'errored'``.
        run_time (str, optional): Wall-clock duration string.
        conformer (int, optional): Conformer index, if applicable.
        tsg (int, optional): TS guess index, if applicable.
        ess_trsh_methods (list, optional): Troubleshooting methods applied.
        timestamp (str, optional): ISO 8601 creation timestamp.
        metadata (dict, optional): Extra data.
    """

    def __init__(self,
                 node_id: str,
                 label: Optional[str] = None,
                 job_name: Optional[str] = None,
                 job_type: Optional[str] = None,
                 job_adapter: Optional[str] = None,
                 level: Optional[str] = None,
                 status: Optional[str] = None,
                 run_time: Optional[str] = None,
                 conformer: Optional[int] = None,
                 tsg: Optional[int] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 timestamp: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__(node_id=node_id, node_type=NodeType.calculation,
                         label=label, timestamp=timestamp, metadata=metadata)
        self.job_name = job_name
        self.job_type = job_type
        self.job_adapter = job_adapter
        self.level = level
        self.status = status
        self.run_time = run_time
        self.conformer = conformer
        self.tsg = tsg
        self.ess_trsh_methods = ess_trsh_methods

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        if self.job_name is not None:
            d['job_name'] = self.job_name
        if self.job_type is not None:
            d['job_type'] = self.job_type
        if self.job_adapter is not None:
            d['job_adapter'] = self.job_adapter
        if self.level is not None:
            d['level'] = self.level
        if self.status is not None:
            d['status'] = self.status
        if self.run_time is not None:
            d['run_time'] = self.run_time
        if self.conformer is not None:
            d['conformer'] = self.conformer
        if self.tsg is not None:
            d['tsg'] = self.tsg
        if self.ess_trsh_methods:
            d['ess_trsh_methods'] = list(self.ess_trsh_methods)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalculationNode':
        obj = object.__new__(cls)
        obj.node_id = d['node_id']
        obj.node_type = d.get('node_type', NodeType.calculation)
        obj.label = d.get('label')
        obj.timestamp = d.get('timestamp')
        obj.metadata = d.get('metadata')
        obj.job_name = d.get('job_name')
        obj.job_type = d.get('job_type')
        obj.job_adapter = d.get('job_adapter')
        obj.level = d.get('level')
        obj.status = d.get('status')
        obj.run_time = d.get('run_time')
        obj.conformer = d.get('conformer')
        obj.tsg = d.get('tsg')
        obj.ess_trsh_methods = d.get('ess_trsh_methods')
        return obj


class DataNode(ProvenanceNode):
    """
    A data artifact node (geometry, energy, frequencies, etc.).

    Args:
        node_id (str): Unique identifier.
        label (str, optional): Species label.
        data_kind (str, optional): One of :class:`DataKind` values.
        value: The scalar or small data payload (energy float, freq list, etc.).
        source_path (str, optional): Path to the file containing this data.
        timestamp (str, optional): ISO 8601 creation timestamp.
        metadata (dict, optional): Extra data.
    """

    def __init__(self,
                 node_id: str,
                 label: Optional[str] = None,
                 data_kind: Optional[str] = None,
                 value: Any = None,
                 source_path: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__(node_id=node_id, node_type=NodeType.data,
                         label=label, timestamp=timestamp, metadata=metadata)
        self.data_kind = _enum_val(data_kind) if data_kind is not None else None
        self.value = value
        self.source_path = source_path

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        if self.data_kind is not None:
            d['data_kind'] = self.data_kind
        if self.value is not None:
            d['value'] = self.value
        if self.source_path is not None:
            d['source_path'] = self.source_path
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataNode':
        obj = object.__new__(cls)
        obj.node_id = d['node_id']
        obj.node_type = d.get('node_type', NodeType.data)
        obj.label = d.get('label')
        obj.timestamp = d.get('timestamp')
        obj.metadata = d.get('metadata')
        obj.data_kind = d.get('data_kind')
        obj.value = d.get('value')
        obj.source_path = d.get('source_path')
        return obj


class DecisionNode(ProvenanceNode):
    """
    An algorithmic decision point (conformer selection, TS validation, etc.).

    Args:
        node_id (str): Unique identifier.
        label (str, optional): Species label.
        decision_kind (str, optional): One of :class:`DecisionKind` values.
        criteria (dict, optional): The selection/rejection criteria applied.
        outcome (str, optional): Human-readable summary of the decision result.
        timestamp (str, optional): ISO 8601 creation timestamp.
        metadata (dict, optional): Extra data.
    """

    def __init__(self,
                 node_id: str,
                 label: Optional[str] = None,
                 decision_kind: Optional[str] = None,
                 criteria: Optional[Dict[str, Any]] = None,
                 outcome: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__(node_id=node_id, node_type=NodeType.decision,
                         label=label, timestamp=timestamp, metadata=metadata)
        self.decision_kind = _enum_val(decision_kind) if decision_kind is not None else None
        self.criteria = criteria
        self.outcome = outcome

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        if self.decision_kind is not None:
            d['decision_kind'] = self.decision_kind
        if self.criteria:
            d['criteria'] = self.criteria
        if self.outcome is not None:
            d['outcome'] = self.outcome
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DecisionNode':
        obj = object.__new__(cls)
        obj.node_id = d['node_id']
        obj.node_type = d.get('node_type', NodeType.decision)
        obj.label = d.get('label')
        obj.timestamp = d.get('timestamp')
        obj.metadata = d.get('metadata')
        obj.decision_kind = d.get('decision_kind')
        obj.criteria = d.get('criteria')
        obj.outcome = d.get('outcome')
        return obj


# ── Edge class ───────────────────────────────────────────────────────────────


class ProvenanceEdge(object):
    """
    A typed directed edge in the provenance DAG.

    Args:
        source_id (str): Node ID of the edge source.
        target_id (str): Node ID of the edge target.
        edge_type (str): One of :class:`EdgeType` values.
        metadata (dict, optional): Arbitrary extra key-value data.
    """

    def __init__(self,
                 source_id: str,
                 target_id: str,
                 edge_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 ):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = _enum_val(edge_type)
        self.metadata = metadata

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
        }
        if self.metadata:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProvenanceEdge':
        obj = object.__new__(cls)
        obj.source_id = d['source_id']
        obj.target_id = d['target_id']
        obj.edge_type = d.get('edge_type', '')
        obj.metadata = d.get('metadata')
        return obj

    def __repr__(self) -> str:
        return f'ProvenanceEdge({self.source_id!r} --{self.edge_type}--> {self.target_id!r})'
