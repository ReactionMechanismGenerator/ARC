"""
ProvenanceGraph — a directed acyclic graph for tracking ARC computational provenance.

The graph stores typed nodes (species, calculations, data artifacts, decisions)
connected by typed directed edges (input_of, selected_by, troubleshot_by, etc.).
It supports forward/backward traversal, flexible queries, and YAML serialization
via the project's standard ``save_yaml_file`` / ``read_yaml_file`` helpers.
"""

import datetime
import re
from collections import deque
from typing import Any, Dict, List, Optional

from arc.common import get_logger, read_yaml_file, save_yaml_file
from arc.provenance.nodes import (
    CalculationNode,
    DataNode,
    DecisionNode,
    NodeType,
    ProvenanceEdge,
    ProvenanceNode,
    _enum_val,
)

logger = get_logger()

SCHEMA_VERSION = 2


class ProvenanceGraph(object):
    """
    A directed acyclic graph for tracking computational provenance.

    Args:
        project (str, optional): The ARC project name.
        run_id (str, optional): Unique run identifier.

    Attributes:
        nodes (Dict[str, ProvenanceNode]): Maps node_id to node.
        edges (List[ProvenanceEdge]): All directed edges.
    """

    def __init__(self,
                 project: Optional[str] = None,
                 run_id: Optional[str] = None,
                 ):
        self.project = project
        self.run_id = run_id or (
            f'{project}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
            if project else None
        )
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self._counter: int = 0

    # ── Node operations ──────────────────────────────────────────────────────

    def _next_id(self, prefix: str) -> str:
        """Generate the next unique node ID with the given prefix."""
        self._counter += 1
        return f'{prefix}_{self._counter}'

    def add_node(self, node: ProvenanceNode) -> str:
        """
        Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            str: The node's ID.
        """
        if node.node_id in self.nodes:
            logger.debug(f'Node {node.node_id!r} already exists in the provenance graph; skipping.')
            return node.node_id
        self.nodes[node.node_id] = node
        return node.node_id

    def add_species_node(self, label: Optional[str] = None, is_ts: bool = False,
                         timestamp: Optional[str] = None) -> str:
        """
        Convenience method to add a species node.

        Args:
            label: Species label (optional).
            is_ts: Whether this is a transition state.
            timestamp: Optional ISO timestamp.

        Returns:
            str: The new node's ID.
        """
        node_id = self._next_id('species')
        metadata = {'is_ts': is_ts} if is_ts else None
        node = ProvenanceNode(node_id=node_id, node_type=NodeType.species,
                              label=label, timestamp=timestamp, metadata=metadata)
        self.add_node(node)
        return node_id

    def add_calculation_node(self, label: Optional[str] = None, **kwargs) -> str:
        """
        Convenience method to add a calculation node.

        Returns:
            str: The new node's ID.
        """
        node_id = self._next_id('calc')
        node = CalculationNode(node_id=node_id, label=label, **kwargs)
        self.add_node(node)
        return node_id

    def add_data_node(self, label: Optional[str] = None, **kwargs) -> str:
        """
        Convenience method to add a data node.

        Returns:
            str: The new node's ID.
        """
        node_id = self._next_id('data')
        node = DataNode(node_id=node_id, label=label, **kwargs)
        self.add_node(node)
        return node_id

    def add_decision_node(self, label: Optional[str] = None, **kwargs) -> str:
        """
        Convenience method to add a decision node.

        Returns:
            str: The new node's ID.
        """
        node_id = self._next_id('decision')
        node = DecisionNode(node_id=node_id, label=label, **kwargs)
        self.add_node(node)
        return node_id

    def get_node(self, node_id: str) -> Optional[ProvenanceNode]:
        """Return the node with the given ID, or None."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str,
                          label: Optional[str] = None) -> List[ProvenanceNode]:
        """Return all nodes of the given type, optionally filtered by label."""
        results = [n for n in self.nodes.values() if n.node_type == _enum_val(node_type)]
        if label is not None:
            results = [n for n in results if n.label == label]
        return results

    def get_nodes_by_label(self, label: str) -> List[ProvenanceNode]:
        """Return all nodes associated with the given species label."""
        return [n for n in self.nodes.values() if n.label == label]

    def find_species_node(self, label: str) -> Optional[str]:
        """Return the node_id of the species node for the given label, or None."""
        for n in self.nodes.values():
            if n.node_type == 'species' and n.label == label:
                return n.node_id
        return None

    def find_calc_node(self, label: str, job_name: str) -> Optional[str]:
        """Return the node_id of a calculation node matching label and job_name, or None."""
        for n in self.nodes.values():
            if (n.node_type == 'calculation'
                    and n.label == label
                    and getattr(n, 'job_name', None) == job_name):
                return n.node_id
        return None

    def update_node(self, node_id: str, **attrs) -> bool:
        """
        Update attributes on an existing node.

        Args:
            node_id: The node to update.
            **attrs: Attribute names and new values.

        Returns:
            bool: True if the node was found and updated.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return False
        for key, value in attrs.items():
            setattr(node, key, value)
        return True

    # ── Edge operations ──────────────────────────────────────────────────────

    def add_edge(self,
                 source_id: str,
                 target_id: str,
                 edge_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 ) -> ProvenanceEdge:
        """
        Add a directed edge between two nodes.

        Logs a warning if source or target node does not exist in the graph,
        but still creates the edge (the node may be added later on restart).

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: One of :class:`EdgeType` values.
            metadata: Optional extra data.

        Returns:
            The created edge.
        """
        if source_id not in self.nodes:
            logger.warning(f'Creating edge from non-existent source node {source_id!r}')
        if target_id not in self.nodes:
            logger.warning(f'Creating edge to non-existent target node {target_id!r}')
        edge = ProvenanceEdge(source_id=source_id, target_id=target_id,
                              edge_type=edge_type, metadata=metadata)
        self.edges.append(edge)
        return edge

    def get_edges_from(self, node_id: str) -> List[ProvenanceEdge]:
        """Return all edges originating from the given node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> List[ProvenanceEdge]:
        """Return all edges pointing to the given node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_edges_by_type(self, edge_type: str) -> List[ProvenanceEdge]:
        """Return all edges of the given type."""
        return [e for e in self.edges if e.edge_type == _enum_val(edge_type)]

    # ── Traversal ────────────────────────────────────────────────────────────

    def descendants(self, node_id: str) -> List[str]:
        """
        Return all node IDs reachable forward from *node_id* (BFS).

        Does not include *node_id* itself.
        """
        visited = set()
        queue = deque()
        for e in self.edges:
            if e.source_id == node_id:
                queue.append(e.target_id)
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            for e in self.edges:
                if e.source_id == nid and e.target_id not in visited:
                    queue.append(e.target_id)
        return list(visited)

    def ancestors(self, node_id: str) -> List[str]:
        """
        Return all node IDs reachable backward from *node_id* (BFS).

        Does not include *node_id* itself.
        """
        visited = set()
        queue = deque()
        for e in self.edges:
            if e.target_id == node_id:
                queue.append(e.source_id)
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            for e in self.edges:
                if e.target_id == nid and e.source_id not in visited:
                    queue.append(e.source_id)
        return list(visited)

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self,
              node_type: Optional[str] = None,
              label: Optional[str] = None,
              decision_kind: Optional[str] = None,
              data_kind: Optional[str] = None,
              status: Optional[str] = None,
              ) -> List[ProvenanceNode]:
        """
        Flexible query over nodes with optional filters.

        All provided filters are ANDed together.

        Args:
            node_type: Filter by NodeType value.
            label: Filter by species label.
            decision_kind: Filter DecisionNodes by DecisionKind value.
            data_kind: Filter DataNodes by DataKind value.
            status: Filter CalculationNodes by job status.

        Returns:
            List of matching nodes.
        """
        results = list(self.nodes.values())
        if node_type is not None:
            results = [n for n in results if n.node_type == _enum_val(node_type)]
        if label is not None:
            results = [n for n in results if n.label == label]
        if decision_kind is not None:
            results = [n for n in results
                       if getattr(n, 'decision_kind', None) == _enum_val(decision_kind)]
        if data_kind is not None:
            results = [n for n in results
                       if getattr(n, 'data_kind', None) == _enum_val(data_kind)]
        if status is not None:
            results = [n for n in results
                       if getattr(n, 'status', None) == status]
        return results

    # ── Serialization ────────────────────────────────────────────────────────

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the full graph to a dict for YAML output."""
        d: Dict[str, Any] = {
            'schema_version': SCHEMA_VERSION,
        }
        if self.project is not None:
            d['project'] = self.project
        if self.run_id is not None:
            d['run_id'] = self.run_id
        d['nodes'] = [node.as_dict() for node in self.nodes.values()]
        d['edges'] = [edge.as_dict() for edge in self.edges]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProvenanceGraph':
        """Reconstruct a ProvenanceGraph from a dict (e.g. loaded from YAML)."""
        obj = object.__new__(cls)
        obj.project = d.get('project')
        obj.run_id = d.get('run_id')
        obj.nodes = {}
        obj.edges = []
        obj._counter = 0
        for node_dict in d.get('nodes', []):
            node = ProvenanceNode.from_dict(node_dict)
            obj.nodes[node.node_id] = node
            # Update counter to avoid ID collisions on restart.
            match = re.search(r'_(\d+)$', node.node_id)
            if match:
                obj._counter = max(obj._counter, int(match.group(1)))
        for edge_dict in d.get('edges', []):
            obj.edges.append(ProvenanceEdge.from_dict(edge_dict))
        return obj

    def save(self, path: str) -> None:
        """Persist the graph to a YAML file."""
        save_yaml_file(path=path, content=self.as_dict())

    @classmethod
    def load(cls, path: str) -> 'ProvenanceGraph':
        """Load a ProvenanceGraph from a YAML file."""
        data = read_yaml_file(path)
        if not isinstance(data, dict):
            raise ValueError(f'Expected a dict in {path}, got {type(data).__name__}')
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (f'ProvenanceGraph(project={self.project!r}, '
                f'nodes={len(self.nodes)}, edges={len(self.edges)})')
