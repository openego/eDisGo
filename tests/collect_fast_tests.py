import subprocess

from pathlib import Path

output_file = Path("ci/fast_tests.txt")
output_file.parent.mkdir(exist_ok=True)

result = subprocess.run(
    [
        "python",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-m",
        "fast",
    ],
    capture_output=True,
    text=True,
    check=False,
)

tests = []

for line in result.stdout.splitlines():
    line = line.strip()

    if not line:
        continue

    if line.startswith("tests/") and "::" in line:
        tests.append(line)

with output_file.open("w", encoding="utf-8") as f:
    f.write("# Fast tests generated automatically\n")
    f.write("# Lines starting with # are ignored\n\n")

    for test in tests:
        f.write(test + "\n")

print(f"Saved {len(tests)} fast tests to {output_file}")
