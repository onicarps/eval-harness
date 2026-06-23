"""Tests for src/rubric.py — RubricTemplate and RubricManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.db import Database
from src.rubric import RubricManager, RubricTemplate


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Create a fresh database with migration v2 applied."""
    db_path = tmp_path / "test.db"
    return Database(db_path)


@pytest.fixture
def manager(db: Database) -> RubricManager:
    return RubricManager(db)


class TestRubricTemplate:
    def test_create_basic(self) -> None:
        t = RubricTemplate(
            template_id="test-1",
            name="Test",
            yaml_content="dimensions:\n- name: foo\n  weight: 1.0\n  description: test\nscoring:\n  scale: 0-1\noutput_format:\n  foo: float\n",
        )
        assert t.template_id == "test-1"
        assert t.name == "Test"
        assert t.is_builtin is False
        assert len(t.dimensions) == 1

    def test_dimensions_parsed(self) -> None:
        yaml_content = (
            "dimensions:\n"
            "- name: faithfulness\n  weight: 0.5\n  description: grounded?\n"
            "- name: task_completion\n  weight: 0.5\n  description: complete?\n"
            "scoring:\n  scale: 0-1\n"
            "output_format:\n  faithfulness: float\n"
        )
        t = RubricTemplate(template_id="x", name="X", yaml_content=yaml_content)
        assert len(t.dimensions) == 2
        assert t.dimensions[0]["name"] == "faithfulness"
        assert t.dimensions[1]["weight"] == 0.5

    def test_scoring_parsed(self) -> None:
        yaml_content = (
            "dimensions:\n- name: foo\n  weight: 1.0\n  description: test\n"
            "scoring:\n  scale: 0-1\n  pass_threshold: 0.7\n"
            "output_format:\n  foo: float\n"
        )
        t = RubricTemplate(template_id="x", name="X", yaml_content=yaml_content)
        assert t.scoring.get("pass_threshold") == 0.7

    def test_validate_valid(self) -> None:
        t = RubricTemplate(
            template_id="x",
            name="X",
            yaml_content=(
                "dimensions:\n- name: foo\n  weight: 1.0\n  description: test\n"
                "scoring:\n  scale: 0-1\n"
                "output_format:\n  foo: float\n"
            ),
        )
        errors = t.validate()
        assert errors == []

    def test_validate_invalid_yaml(self) -> None:
        t = RubricTemplate(template_id="x", name="X", yaml_content=": [invalid")
        errors = t.validate()
        assert len(errors) > 0

    def test_validate_missing_dimensions(self) -> None:
        t = RubricTemplate(
            template_id="x",
            name="X",
            yaml_content="scoring:\n  scale: 0-1\noutput_format:\n  foo: float\n",
        )
        errors = t.validate()
        assert any("dimensions" in e.lower() for e in errors)

    def test_validate_missing_scoring(self) -> None:
        t = RubricTemplate(
            template_id="x",
            name="X",
            yaml_content="dimensions:\n- name: foo\n  weight: 1.0\n  description: test\noutput_format:\n  foo: float\n",
        )
        errors = t.validate()
        assert any("scoring" in e.lower() for e in errors)

    def test_validate_weight_out_of_range(self) -> None:
        t = RubricTemplate(
            template_id="x",
            name="X",
            yaml_content=(
                "dimensions:\n- name: foo\n  weight: 1.5\n  description: test\n"
                "scoring:\n  scale: 0-1\n"
                "output_format:\n  foo: float\n"
            ),
        )
        errors = t.validate()
        assert any("weight" in e.lower() for e in errors)


class TestRubricManager:
    def test_list_templates_seeded(self, manager: RubricManager) -> None:
        templates = manager.list_templates()
        assert len(templates) == 5
        ids = [t.template_id for t in templates]
        assert "faithfulness-v1" in ids
        assert "safety-v1" in ids
        assert "accuracy-v1" in ids
        assert "conciseness-v1" in ids
        assert "custom-v1" in ids

    def test_list_templates_builtin_first(self, manager: RubricManager) -> None:
        templates = manager.list_templates()
        # Built-in templates should come first
        assert templates[0].is_builtin is True

    def test_get_template(self, manager: RubricManager) -> None:
        t = manager.get_template("faithfulness-v1")
        assert t is not None
        assert t.name == "Faithfulness + Task Completion"
        assert t.is_builtin is True

    def test_get_template_not_found(self, manager: RubricManager) -> None:
        t = manager.get_template("nonexistent")
        assert t is None

    def test_create_template(self, manager: RubricManager, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "dimensions:\n- name: my_dim\n  weight: 1.0\n  description: test\n"
            "scoring:\n  scale: 0-1\n"
            "output_format:\n  my_dim: float\n"
        )
        t = manager.create_template("My Custom", yaml_file.read_text())
        assert t.name == "My Custom"
        assert t.is_builtin is False
        assert t.template_id.startswith("custom-")

    def test_create_template_invalid_yaml(self, manager: RubricManager) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            manager.create_template("Bad", ": [invalid")

    def test_create_template_missing_dimensions(self, manager: RubricManager) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            manager.create_template("Bad", "scoring:\n  scale: 0-1\noutput_format:\n  x: float\n")

    def test_delete_custom_template(self, manager: RubricManager, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "dimensions:\n- name: my_dim\n  weight: 1.0\n  description: test\n"
            "scoring:\n  scale: 0-1\n"
            "output_format:\n  my_dim: float\n"
        )
        t = manager.create_template("To Delete", yaml_file.read_text())
        assert manager.delete_template(t.template_id) is True
        assert manager.get_template(t.template_id) is None

    def test_delete_builtin_refused(self, manager: RubricManager) -> None:
        with pytest.raises(ValueError, match="built-in"):
            manager.delete_template("faithfulness-v1")

    def test_delete_not_found(self, manager: RubricManager) -> None:
        assert manager.delete_template("nonexistent") is False
