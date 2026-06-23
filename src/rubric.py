"""Rubric template management for eval-harness.

Provides CRUD operations for rubric templates stored in SQLite.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from src.db import Database

logger = logging.getLogger(__name__)


def _parse_simple_yaml(yaml_str: str) -> dict[str, Any]:
    """Parse a simple YAML subset (lists, mappings, strings, numbers).

    This is a minimal parser for rubric template YAML. For production use,
    install PyYAML and replace with yaml.safe_load().
    """
    import yaml as _yaml
    return _yaml.safe_load(yaml_str) or {}


class RubricTemplate:
    """A rubric template with parsed YAML content."""

    def __init__(
        self,
        template_id: str,
        name: str,
        yaml_content: str,
        is_builtin: bool = False,
        created_at: datetime | None = None,
    ) -> None:
        self.template_id = template_id
        self.name = name
        self.yaml_content = yaml_content
        self.is_builtin = is_builtin
        self.created_at = created_at or datetime.now(UTC)

    @property
    def dimensions(self) -> list[dict[str, Any]]:
        """Parse and return the dimensions from YAML content."""
        try:
            data = _parse_simple_yaml(self.yaml_content)
            return cast(list[dict[str, Any]], data.get("dimensions", []))
        except Exception:
            return []

    @property
    def scoring(self) -> dict[str, Any]:
        """Parse and return the scoring config from YAML content."""
        try:
            data = _parse_simple_yaml(self.yaml_content)
            return cast(dict[str, Any], data.get("scoring", {}))
        except Exception:
            return {}

    def validate(self) -> list[str]:
        """Validate the YAML content. Returns list of error messages."""
        errors: list[str] = []
        try:
            data = _parse_simple_yaml(self.yaml_content)
        except Exception as exc:
            errors.append(f"Invalid YAML: {exc}")
            return errors

        if not isinstance(data, dict):
            errors.append("YAML content must be a mapping")
            return errors

        if "dimensions" not in data:
            errors.append("Missing 'dimensions' key")
        elif not isinstance(data["dimensions"], list):
            errors.append("'dimensions' must be a list")
        elif len(data["dimensions"]) == 0:
            errors.append("'dimensions' must not be empty")
        else:
            for i, dim in enumerate(data["dimensions"]):
                if not isinstance(dim, dict):
                    errors.append(f"Dimension {i} must be a mapping")
                    continue
                for key in ("name", "weight", "description"):
                    if key not in dim:
                        errors.append(f"Dimension {i} missing '{key}'")
                if "weight" in dim:
                    w = dim["weight"]
                    if not isinstance(w, (int, float)) or w < 0 or w > 1:
                        errors.append(f"Dimension {i} weight must be a number in [0, 1]")

        if "scoring" not in data:
            errors.append("Missing 'scoring' key")

        if "output_format" not in data:
            errors.append("Missing 'output_format' key")

        return errors


class RubricManager:
    """Manages rubric templates in the database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def list_templates(self) -> list[RubricTemplate]:
        """Return all rubric templates, built-in first."""
        cur = self.db.connection.execute(
            "SELECT template_id, name, yaml_content, is_builtin, created_at "
            "FROM rubric_templates ORDER BY is_builtin DESC, name;"
        )
        templates: list[RubricTemplate] = []
        for row in cur.fetchall():
            created_at = _parse_iso(row[4])
            templates.append(
                RubricTemplate(
                    template_id=row[0],
                    name=row[1],
                    yaml_content=row[2],
                    is_builtin=bool(row[3]),
                    created_at=created_at,
                )
            )
        return templates

    def get_template(self, template_id: str) -> RubricTemplate | None:
        """Return a single rubric template by ID."""
        cur = self.db.connection.execute(
            "SELECT template_id, name, yaml_content, is_builtin, created_at "
            "FROM rubric_templates WHERE template_id = ?;",
            (template_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return RubricTemplate(
            template_id=row[0],
            name=row[1],
            yaml_content=row[2],
            is_builtin=bool(row[3]),
            created_at=_parse_iso(row[4]),
        )

    def create_template(self, name: str, yaml_content: str) -> RubricTemplate:
        """Create a new rubric template. Returns the created template."""
        import uuid
        template_id = f"custom-{uuid.uuid4().hex[:8]}"
        template = RubricTemplate(
            template_id=template_id,
            name=name,
            yaml_content=yaml_content,
            is_builtin=False,
        )
        errors = template.validate()
        if errors:
            raise ValueError(f"Invalid rubric template: {'; '.join(errors)}")

        now = datetime.now(UTC).isoformat()
        self.db.connection.execute(
            "INSERT INTO rubric_templates (template_id, name, yaml_content, is_builtin, created_at) "
            "VALUES (?, ?, ?, 0, ?);",
            (template_id, name, yaml_content, now),
        )
        self.db.connection.commit()
        logger.info("Created rubric template %s (%s)", template_id, name)
        return template

    def delete_template(self, template_id: str) -> bool:
        """Delete a rubric template. Returns True if deleted. Refuses built-in."""
        cur = self.db.connection.execute(
            "SELECT is_builtin FROM rubric_templates WHERE template_id = ?;",
            (template_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        if row[0]:
            raise ValueError(f"Cannot delete built-in template: {template_id}")
        self.db.connection.execute(
            "DELETE FROM rubric_templates WHERE template_id = ?;",
            (template_id,),
        )
        self.db.connection.commit()
        logger.info("Deleted rubric template %s", template_id)
        return True


def _parse_iso(s: str | None) -> datetime | None:
    """Parse ISO-8601 string back to datetime; return None for empty input."""
    return datetime.fromisoformat(s) if s else None
