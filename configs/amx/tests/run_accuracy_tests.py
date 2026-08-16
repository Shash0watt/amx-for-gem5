#!/usr/bin/env python3

import argparse
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEM5 = ROOT / "build/X86/gem5.opt"
CONFIG = ROOT / "configs/amx/tb.py"
ACCURACY_BIN_DIR = ROOT / "configs/amx/binaries/accuracy"
DEBUG_DIR = ROOT / "amx_debug"
ACTUAL_DIR = DEBUG_DIR / "actual"
CORRECT_DIR = DEBUG_DIR / "correct"
M5OUT_DIR = DEBUG_DIR / "m5out"

VOLATILE_PREFIXES = (
    "  Tick        :",
    "  Cycle       :",
    "  Accelerator :",
)


def comparable_lines(path):
    return [
        f"{line.rstrip()}\n"
        for line in path.read_text().splitlines()
        if not line.startswith(VOLATILE_PREFIXES)
    ]


def run_make():
    print("Running make all...")
    result = subprocess.run(["make", "all"], cwd=ROOT)
    if result.returncode != 0:
        print("Build failed.")
        sys.exit(1)


def get_test_binaries():
    if not ACCURACY_BIN_DIR.is_dir():
        return []
    return sorted([p for p in ACCURACY_BIN_DIR.iterdir() if p.is_file()])


def run_test(binary):
    dump_name = binary.name[:-5] if binary.name.endswith("_test") else binary.name
    dump = ACTUAL_DIR / f"{dump_name}.txt"
    log = ACTUAL_DIR / f"{dump_name}.log"
    output_dir = M5OUT_DIR / dump_name

    dump.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(GEM5),
        "-d",
        str(output_dir),
        str(CONFIG),
        "--binary",
        str(binary),
        "--dump-directory",
        str(ACTUAL_DIR),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write_text(result.stdout)

    if result.returncode != 0:
        print(f"FAIL {dump_name}: gem5 exited with {result.returncode}")
        print(f"  log: {log.relative_to(ROOT)}")
        return None

    if not dump.is_file():
        alt_dump = ACTUAL_DIR / f"{binary.name}.txt"
        if alt_dump.is_file():
            dump = alt_dump
        else:
            print(f"FAIL {dump_name}: gem5 did not create state dump ({dump.name})")
            print(f"  log: {log.relative_to(ROOT)}")
            return None

    return dump


def compare_dump(actual, correct):
    actual_lines = comparable_lines(actual)
    correct_lines = comparable_lines(correct)
    if actual_lines == correct_lines:
        return True

    diff = difflib.unified_diff(
        correct_lines,
        actual_lines,
        fromfile=str(correct.relative_to(ROOT)),
        tofile=str(actual.relative_to(ROOT)),
    )
    print("".join(diff), end="")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run AMX accuracy tests")
    parser.add_argument(
        "--update-correct",
        action="store_true",
        help="replace correct dumps with current simulator output",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip make all before running tests",
    )
    args = parser.parse_args()

    if not GEM5.is_file():
        print(f"Missing gem5 binary: {GEM5}")
        return 1

    # 1. Run make to compile tests into configs/amx/binaries/
    if not args.no_build:
        run_make()

    binaries = get_test_binaries()
    if not binaries:
        print(f"No test binaries found in {ACCURACY_BIN_DIR.relative_to(ROOT)}")
        return 1

    ACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    CORRECT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Run simulation and 3. Compare with correct dumps
    failures = 0
    for binary in binaries:
        dump_name = binary.name[:-5] if binary.name.endswith("_test") else binary.name
        actual = run_test(binary)
        if actual is None:
            failures += 1
            continue

        correct = CORRECT_DIR / actual.name
        if args.update_correct:
            correct.write_text("".join(comparable_lines(actual)))
            print(f"UPDATED {dump_name}")
        elif not correct.is_file():
            print(f"FAIL {dump_name}: missing reference dump {correct.relative_to(ROOT)}")
            failures += 1
        elif compare_dump(actual, correct):
            print(f"PASS {dump_name}")
        else:
            print(f"FAIL {dump_name}: state differs from correct dump")
            failures += 1

    if failures:
        print(f"\n{failures} AMX accuracy test(s) failed")
        return 1

    action = "updated" if args.update_correct else "passed"
    print(f"\nAll {len(binaries)} AMX accuracy tests {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


