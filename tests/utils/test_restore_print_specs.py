"""Characterization test for the spec listing the restore CLIs print."""

from __future__ import annotations

from unittest.mock import Mock

from cuvis_ai_core.utils.restore import _print_specs


def test_print_specs_lists_inputs_then_outputs(capsys) -> None:
    pipeline = Mock()
    pipeline.get_input_specs.return_value = {"cube": "spec-a", "mask": "spec-b"}
    pipeline.get_output_specs.return_value = {"scores": "spec-c"}

    _print_specs(pipeline)

    assert capsys.readouterr().out == (
        "\nInput Specs:\n  cube: spec-a\n  mask: spec-b\n\nOutput Specs:\n  scores: spec-c\n"
    )
