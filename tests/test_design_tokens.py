import re
from pathlib import Path

ALLOWED_HEX = {
    "#111111",
    "#000000",
    "#540000",
    "#3a0101",
    "#004749",
    "#013b3f",
    "#b09b72",
    "#938160",
    "#ededed",
    "#cccccc",
}

ALLOWED_RGB_TUPLES = {
    (17, 17, 17),
    (0, 0, 0),
    (84, 0, 0),
    (58, 1, 1),
    (0, 71, 73),
    (1, 59, 63),
    (176, 155, 114),
    (147, 129, 96),
    (237, 237, 237),
    (204, 204, 204),
}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "vendor",
    "data",
    ".ruff_cache",
    ".pytest_cache",
    ".system_generated",
    "reports",
}

def is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    return False

def check_file_colors(path: Path) -> list[str]:
    violations: list[str] = []
    content = path.read_text(encoding="utf-8")
    hex_pattern = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    for match in hex_pattern.finditer(content):
        val = match.group(0).lower()
        if val not in ALLOWED_HEX:
            violations.append(f"{path}: disallowed hex color '{val}'")

    rgba_pattern = re.compile(r"rgba\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([0-9.]+)\s*\)")
    for match in rgba_pattern.finditer(content):
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if (r, g, b) not in ALLOWED_RGB_TUPLES:
            violations.append(f"{path}: disallowed rgba base channels ({r}, {g}, {b})")

    rgb_pattern = re.compile(r"(?<!a)\brgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
    for match in rgb_pattern.finditer(content):
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if (r, g, b) not in ALLOWED_RGB_TUPLES:
            violations.append(f"{path}: disallowed rgb channels ({r}, {g}, {b})")

    return violations

def test_design_tokens_palette() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "loop" / "web"
    violations: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path):
            continue
        if path.suffix in {".css", ".js", ".mjs"}:
            violations.extend(check_file_colors(path))

    assert not violations, "\n".join(violations)
