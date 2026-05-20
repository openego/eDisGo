from pathlib import Path

all_tests_file = Path("all_tests.txt")
fast_tests_file = Path("ci/fast_tests.txt")
output_file = Path("ci/non_fast_tests.txt")


def load_tests(file_path):
    if not file_path.exists():
        return set()

    tests = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        tests.add(line)

    return tests


all_tests = load_tests(all_tests_file)
fast_tests = load_tests(fast_tests_file)

# quitar fast de todos
non_fast_tests = sorted(all_tests - fast_tests)

output_file.parent.mkdir(exist_ok=True)

with output_file.open("w", encoding="utf-8") as f:
    f.write("# Non-fast tests (auto-generated)\n\n")
    for test in non_fast_tests:
        f.write(test + "\n")

print(f"Total tests: {len(all_tests)}")
print(f"Fast tests: {len(fast_tests)}")
print(f"Non-fast tests: {len(non_fast_tests)}")
print(f"Saved to: {output_file}")
