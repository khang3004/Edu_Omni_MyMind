"""Neo4j client wrapper implementing the GraphStore interface.

Handles thread-safe database connections, auto-indexing, and Cypher queries.
"""

from __future__ import annotations

import re
from typing import Any

from edumind.core.exceptions import GraphStoreError
from edumind.core.logging import get_logger
from edumind.services.graphstore.base import GraphStore
from edumind.utils.retry import retry_on_transient_error

logger = get_logger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j driver wrapper for Knowledge Graph operations."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "omni_ips_password",
    ):
        """Initializes connection parameters and attempts driver setup."""
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None
        self._ready = False

        self.connect()

    def connect(self) -> None:
        """Initializes the Neo4j GraphDatabase driver and runs a quick verification query."""
        try:
            from neo4j import GraphDatabase

            logger.info("connecting_to_neo4j", uri=self._uri, user=self._user)
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            # Verify connection
            self._driver.verify_connectivity()
            self._ready = True
            self._ensure_constraints()
            logger.info("neo4j_graphstore_ready")
        except Exception as e:
            logger.warning("neo4j_connection_failed_bypassing", error=str(e))
            self._ready = False
            self._driver = None

    def _ensure_constraints(self) -> None:
        """Creates indexes and unique constraints on the Concept node label to optimize matches."""
        if not self.is_ready:
            return
        try:
            with self._driver.session() as session:
                # Create constraint if not exists (syntax varies by neo4j version, using modern compatible syntax)
                session.run(
                    "CREATE CONSTRAINT unique_concept_name IF NOT EXISTS "
                    "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
                )
                logger.info("neo4j_constraints_ensured")
        except Exception as e:
            logger.warning("neo4j_constraints_creation_warning", error=str(e))

    @property
    def is_ready(self) -> bool:
        """True if driver is connected and active."""
        return self._ready and self._driver is not None

    def _sanitize(self, value: str) -> str:
        """Sanitizes labels/relationships to prevent Cypher injection."""
        return re.sub(r"[^a-zA-Z0-9_]", "", value)

    @retry_on_transient_error(max_attempts=3)
    def upsert_entity(
        self,
        name: str,
        entity_type: str = "Concept",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Creates or updates a concept node in Neo4j."""
        if not self.is_ready:
            logger.debug("neo4j_not_ready_skipping_upsert_entity", name=name)
            return

        normalized_name = name.strip()
        if not normalized_name:
            return

        props = properties or {}
        # Ensure name is in properties
        props["name"] = normalized_name
        props["entity_type"] = entity_type

        # We use a base label :Concept for all Graph RAG nodes to ensure fast indices,
        # but we also set a dynamic secondary label for categorization
        sanitized_type = self._sanitize(entity_type)
        if not sanitized_type:
            sanitized_type = "Concept"

        query = f"MERGE (c:Concept {{name: $name}}) SET c:{sanitized_type} SET c += $properties"

        try:
            with self._driver.session() as session:
                session.run(query, name=normalized_name, properties=props)
                logger.debug(
                    "neo4j_upsert_entity_completed", name=normalized_name, type=entity_type
                )
        except Exception as e:
            logger.error("neo4j_upsert_entity_failed", name=normalized_name, error=str(e))
            raise GraphStoreError(
                f"Failed to upsert entity {normalized_name} in Neo4j",
                details={"name": normalized_name, "error": str(e)},
            ) from e

    @retry_on_transient_error(max_attempts=3)
    def upsert_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Links two concepts with a directed relationship, creating missing nodes if necessary."""
        if not self.is_ready:
            logger.debug(
                "neo4j_not_ready_skipping_upsert_relationship", source=source, target=target
            )
            return

        src = source.strip()
        tgt = target.strip()
        if not src or not tgt:
            return

        props = properties or {}
        sanitized_rel = self._sanitize(rel_type)
        if not sanitized_rel:
            sanitized_rel = "RELATED_TO"

        # Create nodes if missing first to avoid Match failures
        query_nodes = "MERGE (a:Concept {name: $source}) MERGE (b:Concept {name: $target})"

        query_rel = (
            "MATCH (a:Concept {name: $source}), (b:Concept {name: $target}) "
            f"MERGE (a)-[r:{sanitized_rel}]->(b) "
            "SET r += $properties"
        )

        try:
            with self._driver.session() as session:
                session.run(query_nodes, source=src, target=tgt)
                session.run(query_rel, source=src, target=tgt, properties=props)
                logger.debug(
                    "neo4j_upsert_relationship_completed", source=src, target=tgt, type=rel_type
                )
        except Exception as e:
            logger.error("neo4j_upsert_relationship_failed", source=src, target=tgt, error=str(e))
            raise GraphStoreError(
                f"Failed to upsert relationship between {src} and {tgt}",
                details={"source": src, "target": tgt, "rel_type": rel_type, "error": str(e)},
            ) from e

    def query_neighborhood(self, entity_name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Finds neighboring concepts and their edge weights/relationships in Neo4j."""
        if not self.is_ready:
            return []

        name = entity_name.strip()
        # Neo4j Cypher to retrieve connected nodes up to dynamic depth
        query = (
            "MATCH (a:Concept {name: $name})-[r]-(b:Concept) "
            "RETURN a.name as source, type(r) as relationship, b.name as target, "
            "properties(b) as target_properties, properties(r) as relationship_properties "
            "LIMIT 50"
        )

        try:
            with self._driver.session() as session:
                result = session.run(query, name=name)
                records = []
                for record in result:
                    records.append(
                        {
                            "source": record["source"],
                            "relationship": record["relationship"],
                            "target": record["target"],
                            "target_type": record["target_properties"].get(
                                "entity_type", "Concept"
                            ),
                            "target_properties": record["target_properties"],
                            "relationship_properties": record["relationship_properties"],
                        }
                    )
                return records
        except Exception as e:
            logger.error("neo4j_query_neighborhood_failed", name=name, error=str(e))
            return []

    def clear_graph(self) -> bool:
        """Deletes all nodes and relations in the active Neo4j database."""
        if not self.is_ready:
            return False

        query = "MATCH (n) DETACH DELETE n"
        try:
            with self._driver.session() as session:
                session.run(query)
                logger.warning("neo4j_database_completely_wiped")
                return True
        except Exception as e:
            logger.error("neo4j_clear_graph_failed", error=str(e))
            return False

    def graph_info(self) -> dict[str, Any]:
        """Counts nodes, labels, and edges inside the graph."""
        if not self.is_ready:
            return {"status": "not_ready", "nodes_count": 0, "edges_count": 0}

        try:
            with self._driver.session() as session:
                nodes_res = session.run("MATCH (n:Concept) RETURN count(n) as count")
                nodes_count = nodes_res.single()["count"]

                edges_res = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                edges_count = edges_res.single()["count"]

                return {
                    "status": "ready",
                    "storage_mode": "neo4j_server",
                    "nodes_count": nodes_count,
                    "edges_count": edges_count,
                }
        except Exception as e:
            logger.error("neo4j_graph_info_failed", error=str(e))
            return {"status": "error", "error": str(e), "nodes_count": 0, "edges_count": 0}

    def close(self) -> None:
        """Closes the underlying Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._ready = False
            logger.info("neo4j_driver_closed")
