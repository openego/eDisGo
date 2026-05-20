import subprocess
import sys

from pathlib import Path

test_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ci/non_fast_tests.txt")


# test_file = Path("ci/non_fast_tests.txt")

if not test_file.exists():
    print(f"ERROR: {test_file} does not exist")
    sys.exit(1)

tests = []

for line in test_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()

    if not line:
        continue

    if line.startswith("#"):
        continue

    tests.append(line)

if not tests:
    print("No active tests found in ci/non_fast_tests.txt")
    sys.exit(1)

cmd = [
    "python",
    "-m",
    "pytest",
    *tests,
    "-vv",
    "--html=non_fast-report.html",
    "--self-contained-html",
    "--junitxml=non_fast-report.xml",
]

print("Running tests:")
for test in tests:
    print(f" - {test}")

result = subprocess.run(cmd)
sys.exit(result.returncode)
