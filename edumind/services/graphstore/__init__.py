"""EduMIND Knowledge Graph Database services package.

Exports the base interface and standard providers (Neo4j, Mock).
"""

from __future__ import annotations

from edumind.services.graphstore.base import GraphStore
from edumind.services.graphstore.mock import MockGraphStore
from edumind.services.graphstore.neo4j import Neo4jGraphStore

__all__ = [
    "GraphStore",
    "MockGraphStore",
    "Neo4jGraphStore",
]
