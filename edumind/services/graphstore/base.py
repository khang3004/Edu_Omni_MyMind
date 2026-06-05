"""Base interface for Knowledge Graph Store service in EduMIND.

Defines standard entity-relationship operations to store and query context.
"""

from __future__ import annotations

from typing import Any, Protocol


class GraphStore(Protocol):
    """Protocol defining core knowledge graph operations for RAG context extraction."""

    def connect(self) -> None:
        """Establishes a connection to the graph database."""
        ...

    @property
    def is_ready(self) -> bool:
        """Checks if the connection to the graph database is active."""
        ...

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "Concept",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Creates or updates an entity (node) in the graph.

        Args:
            name: The unique name identifier of the entity.
            entity_type: The label/type of the entity (e.g. Concept, Term, Person).
            properties: Additional attributes to store on the node.
        """
        ...

    def upsert_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Creates or updates a relationship (edge) between two entities in the graph.

        Args:
            source: Name of the source entity.
            target: Name of the target entity.
            rel_type: Type of the relationship (e.g. DEFINES, PART_OF, PREREQUISITE_FOR).
            properties: Additional attributes to store on the relationship.
        """
        ...

    def query_neighborhood(self, entity_name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Retrieves nodes and relationships connected to the target entity.

        Args:
            entity_name: Name of the entity to start traversal.
            depth: Traversal depth limit (default: 1).

        Returns:
            A list of dictionary mappings representing neighbors and relationship types.
        """
        ...

    def clear_graph(self) -> bool:
        """Deletes all nodes and relationships in the graph database.

        Returns:
            True if wiped successfully, False otherwise.
        """
        ...

    def graph_info(self) -> dict[str, Any]:
        """Gathers graph statistics such as node count, edge count, etc.

        Returns:
            A dictionary containing stats.
        """
        ...
