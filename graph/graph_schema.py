"""
Sentinel-Graph Engine — Graph Schema
Defines node and edge types for the heterogeneous identity knowledge graph.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any


class NodeType(Enum):
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    RESOURCE = "resource"


class EdgeType(Enum):
    USER_HAS_ROLE = "user_has_role"
    ROLE_GRANTS_PERMISSION = "role_grants_permission"
    PERMISSION_ACCESSES_RESOURCE = "permission_accesses_resource"
    USER_ACCESSED_RESOURCE = "user_accessed_resource"  # Behavioral (dynamic)


# Node attribute schemas
USER_ATTRS = [
    "department", "seniority", "office_city", "office_lat", "office_lon",
    "hire_date", "is_active",
]

ROLE_ATTRS = [
    "role_name", "department", "seniority_level",
]

RESOURCE_ATTRS = [
    "resource_name", "sensitivity", "data_type",
]

PERMISSION_ATTRS = [
    "role_id", "resource_id", "action",
]

# Feature encoding maps for GNN
DEPARTMENT_ENCODING = {
    "Engineering": 0, "Finance": 1, "Human Resources": 2, "Marketing": 3,
    "Sales": 4, "IT Operations": 5, "Legal": 6, "Executive": 7,
}

SENIORITY_ENCODING = {
    "Junior": 0, "Junior-Mid": 0, "Mid": 1, "Senior": 2,
    "Senior-Lead": 2, "Lead": 3, "Director": 4, "Director+": 4,
}

SENSITIVITY_ENCODING = {
    "Low": 0, "Medium": 1, "High": 2, "Critical": 3,
}

ACTION_ENCODING = {
    "read": 0, "write": 1, "admin": 2,
}
