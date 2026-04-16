from __future__ import annotations

import re


_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_variables(template: str) -> list[str]:
    return list(dict.fromkeys(_VAR_PATTERN.findall(template)))


def render_template(template: str, variables: dict[str, str]) -> str:
    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Missing template variable: {key}")
        return str(variables[key])

    return _VAR_PATTERN.sub(_replacer, template)


def expand_sweep(
    template: str, variable_sets: list[dict[str, str]]
) -> list[str]:
    return [render_template(template, vs) for vs in variable_sets]
