"""Mock in-memory implementation of the GraphStore interface.

Useful for unit testing and local development without a Neo4j server instance.
"""

from __future__ import annotations

from typing import Any

from edumind.core.logging import get_logger
from edumind.services.graphstore.base import GraphStore

logger = get_logger(__name__)


class MockGraphStore(GraphStore):
    """In-memory thread-safe mock graph database."""

    def __init__(self) -> None:
        """Initializes empty nodes and edges stores."""
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._ready = True
        logger.info("mock_graph_store_initialized")

    def connect(self) -> None:
        """Simulates connection establish."""
        self._ready = True

    @property
    def is_ready(self) -> bool:
        """Always ready as it runs in local RAM."""
        return self._ready

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "Concept",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Stores entity details inside local dictionary."""
        props = properties or {}
        normalized_name = name.strip()
        if not normalized_name:
            return

        self._nodes[normalized_name] = {
            "name": normalized_name,
            "type": entity_type,
            "properties": props,
        }
        logger.debug("mock_graph_upsert_entity", name=normalized_name, type=entity_type)

    def upsert_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Stores a relationship. Ensures endpoints exist as placeholder entities."""
        props = properties or {}
        src = source.strip()
        tgt = target.strip()
        if not src or not tgt:
            return

        # Ensure source and target nodes exist in the nodes store
        if src not in self._nodes:
            self.upsert_entity(src, "Concept")
        if tgt not in self._nodes:
            self.upsert_entity(tgt, "Concept")

        # Check if relationship already exists
        exists = False
        for edge in self._edges:
            if edge["source"] == src and edge["target"] == tgt and edge["type"] == rel_type:
                edge["properties"].update(props)
                exists = True
                break

        if not exists:
            self._edges.append({
                "source": src,
                "target": tgt,
                "type": rel_type,
                "properties": props,
            })
        logger.debug("mock_graph_upsert_relationship", source=src, target=tgt, type=rel_type)

    def query_neighborhood(self, entity_name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Retrieves nodes and edges directly connected to the starting entity name."""
        name = entity_name.strip()
        if name not in self._nodes:
            return []

        results = []
        # Support simple 1-hop traversal for mock search
        for edge in self._edges:
            if edge["source"] == name:
                target_node = self._nodes.get(edge["target"])
                if target_node:
                    results.append({
                        "source": name,
                        "relationship": edge["type"],
                        "target": edge["target"],
                        "target_type": target_node["type"],
                        "target_properties": target_node["properties"],
                        "relationship_properties": edge["properties"],
                    })
            elif edge["target"] == name:
                source_node = self._nodes.get(edge["source"])
                if source_node:
                    results.append({
                        "source": edge["source"],
                        "relationship": edge["type"],
                        "target": name,
                        "source_type": source_node["type"],
                        "source_properties": source_node["properties"],
                        "relationship_properties": edge["properties"],
                    })
        return results

    def clear_graph(self) -> bool:
        """Clears all in-memory data tables."""
        self._nodes.clear()
        self._edges.clear()
        logger.warning("mock_graph_cleared")
        return True

    def graph_info(self) -> dict[str, Any]:
        """Returns statistical counts."""
        return {
            "status": "ready",
            "storage_mode": "mock_memory",
            "nodes_count": len(self._nodes),
            "edges_count": len(self._edges),
        }
