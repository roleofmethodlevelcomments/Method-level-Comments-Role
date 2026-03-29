#!/usr/bin/env python3
"""
=============================================================================
Assertion Effectiveness Evaluation for ASE-Style LLM Oracle Research
=============================================================================

ASSUMPTIONS
-----------
1. Each evaluation instance corresponds to one bug-fix pair and one focal test
   method. The test method has an assertion-free body; you insert generated
   assertions and run the test against buggy and fixed code.

2. Buggy and fixed versions are available either as:
   (a) Separate project directories (--project-paths JSON), or
   (b) Via commit-based materialization: --repos JSON maps project_name to
   repo_url; the script clones and checks out bug_id^ (buggy) and bug_id (fixed)
   using each entry's bug_report.bug_id (commit hash) and project_name.

3. Generated assertions are plain Java assertion statements (e.g. JUnit
   assertEquals, assertTrue, assertThrows). They are inserted into the
   assertion-free test body before the method's closing brace.

4. Compilation and test execution use Maven or Gradle. The script writes the
   test class (with inserted assertion) to the project's test source tree and
   runs the build tool. You can replace this with custom commands.

5. Effectiveness is defined by: compiles, fails on buggy, passes on fixed.
   Optionally, coverage (line, method, branch) is collected as secondary
   metrics when --coverage is set and JaCoCo is configured in the project.

6. Pass/fail is determined by parsing Maven output (BUILD SUCCESS, Failures: 0),
   not by return code alone. RUNTIME_ERROR is used only for invalid execution
   (timeout, JVM crash, OOM), not for normal assertion failure or assertThrows.
   Test isolation: mvn clean test and a unique test class per (bug_id, mode).

ADAPTING PROJECT-SPECIFIC COMPILE/TEST COMMANDS
-----------------------------------------------
- Set PROJECT_PATHS (or load from JSON) to map (project_name, bug_id) or
  project_name to (buggy_dir, fixed_dir). If unset, the script can still
  merge data and produce per-instance results with outcome NOT_EVALUATED.

- Override compile_test(), run_test(), and run_test_with_coverage() in class
  ProjectRunner to use your build system (Maven/Gradle) and environment.
  With --coverage, JaCoCo is used (mvn jacoco:prepare-agent test jacoco:report);
  report is read from target/site/jacoco/jacoco.xml.

- Test class name: if your dataset does not include test_class_name, set
  TEST_CLASS_FROM_ENTRY or derive from test_file_path / project layout.

Usage (merged input):
  python evaluate_assertion_effectiveness.py --input evaluation_merged.json --output-dir results

Usage (dataset + oracle files, merged by bug_id):
  python evaluate_assertion_effectiveness.py --dataset strengthened_comments_full.json \\
    --oracles mode1:llm_generated_oracles_step4_without_comments_deepseek_hybrid_balanced.json \\
                 mode2:llm_generated_oracles_step5_with_comments_deepseek_hybrid_balanced.json \\
    --output-dir results

Author: Research pipeline; defensible for top-tier SE venues.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Outcome categories (semantic oracle effectiveness)
# -----------------------------------------------------------------------------
OUTCOME_EFFECTIVE = "EFFECTIVE"
OUTCOME_INEFFECTIVE_PASS_BOTH = "INEFFECTIVE_PASS_BOTH"
OUTCOME_INEFFECTIVE_FAIL_BOTH = "INEFFECTIVE_FAIL_BOTH"
OUTCOME_INEFFECTIVE_PASS_BUGGY_FAIL_FIXED = "INEFFECTIVE_PASS_BUGGY_FAIL_FIXED"
OUTCOME_COMPILE_ERROR = "COMPILE_ERROR"
OUTCOME_RUNTIME_ERROR = "RUNTIME_ERROR"
OUTCOME_NOT_EVALUATED = "NOT_EVALUATED"
OUTCOME_EMPTY_ASSERTION = "EMPTY_ASSERTION"

ALL_OUTCOMES = [
    OUTCOME_EFFECTIVE,
    OUTCOME_INEFFECTIVE_PASS_BOTH,
    OUTCOME_INEFFECTIVE_FAIL_BOTH,
    OUTCOME_INEFFECTIVE_PASS_BUGGY_FAIL_FIXED,
    OUTCOME_COMPILE_ERROR,
    OUTCOME_RUNTIME_ERROR,
    OUTCOME_NOT_EVALUATED,
    OUTCOME_EMPTY_ASSERTION,
]


def _outcome_label(o: str) -> str:
    return o


# -----------------------------------------------------------------------------
# Maven output parsing (ASE-level rigor: do not rely on return code alone)
# -----------------------------------------------------------------------------
def _parse_maven_test_result(output: str) -> Tuple[bool, bool]:
    """
    Parse Maven/Surefire output to determine test result and whether execution was invalid.
    Returns (passed, invalid_execution).
    - passed: True iff BUILD SUCCESS and test run reports 0 failures.
    - invalid_execution: True only for JVM crash, timeout, OOM, or other invalid run
      (not for normal assertion failure or assertThrows).
    """
    out = (output or "").strip()
    # Build outcome: prefer explicit BUILD SUCCESS / BUILD FAILURE
    build_success = "BUILD SUCCESS" in out
    build_failure = "BUILD FAILURE" in out
    # Test failures: Surefire reports "Tests run: N, Failures: M" or "Failures: M"
    failures_match = re.search(r"Failures:\s*(\d+)", out)
    failures = int(failures_match.group(1)) if failures_match else (0 if build_success else 1)
    errors_match = re.search(r"Errors:\s*(\d+)", out)
    errors = int(errors_match.group(1)) if errors_match else 0
    # Passed: build succeeded and no test failures/errors
    passed = build_success and failures == 0 and errors == 0
    if not passed and not build_success and not build_failure:
        # No clear build status (e.g. truncated output); fall back to returncode elsewhere
        passed = False
    # Invalid execution: timeout, JVM crash, OOM — not normal assertion failure
    invalid_execution = False
    invalid_patterns = [
        r"Run timeout",
        r"Timeout",
        r"OutOfMemoryError",
        r"java\.lang\.OutOfMemoryError",
        r"InternalError",
        r"java\.lang\.InternalError",
        r"FATAL",
        r"JVM crash",
        r"Unable to execute",
    ]
    for pat in invalid_patterns:
        if re.search(pat, out, re.IGNORECASE):
            invalid_execution = True
            break
    return passed, invalid_execution


def _sanitize_for_class_name(s: str, max_len: int = 24) -> str:
    """Sanitize string for use in Java class name (alphanumeric + underscore)."""
    s = re.sub(r"[^a-zA-Z0-9]", "_", (s or "").strip())
    s = s.strip("_") or "Gen"
    return s[:max_len] if len(s) > max_len else s


# -----------------------------------------------------------------------------
# Input: field names for flexible schema
# -----------------------------------------------------------------------------
def get_bug_id(entry: Dict[str, Any]) -> str:
    """Extract bug_id from entry (dataset or merged)."""
    bid = entry.get("bug_id")
    if bid is not None:
        return str(bid).strip()
    br = entry.get("bug_report") or {}
    return str(br.get("bug_id", "")).strip()


def get_project_name(entry: Dict[str, Any]) -> str:
    """Extract project name."""
    p = entry.get("project_name")
    if p is not None:
        return str(p).strip()
    br = entry.get("bug_report") or {}
    return str(br.get("project_name", "")).strip()


def get_assertion_free_test(entry: Dict[str, Any]) -> str:
    """Get the test method body without assertions (where to insert oracles)."""
    s = (
        entry.get("assertion_free_test_code")
        or entry.get("test_case_without_assertions")
        or ""
    )
    return s.strip() if s else ""


def get_generated_assertions_by_mode(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Get generated assertions per mode. Keys: mode names (e.g. code_only, original_comment).
    Values: raw assertion text (may be multi-line).
    """
    by_mode = entry.get("generated_assertions_by_mode")
    if isinstance(by_mode, dict):
        return {k: (v or "").strip() for k, v in by_mode.items()}
    return {}


def get_test_method_name(entry: Dict[str, Any]) -> Optional[str]:
    """Infer or get test method name from entry."""
    name = entry.get("test_method_name")
    if name:
        return str(name).strip()
    test_code = get_assertion_free_test(entry)
    if not test_code:
        return None
    # e.g. "public void testgetMaxVersionsBugFix() {"
    m = re.search(r"\bvoid\s+(\w+)\s*\(", test_code)
    return m.group(1) if m else None


def get_test_class_name(entry: Dict[str, Any]) -> Optional[str]:
    """Get test class name from entry or infer from file path."""
    name = entry.get("test_class_name")
    if name:
        return str(name).strip()
    path = entry.get("test_file_path") or (entry.get("bug_report") or {}).get("file_path")
    if path:
        # e.g. src/test/java/.../HColumnDescriptorTest.java -> HColumnDescriptorTest
        base = Path(path).stem
        if base.endswith("Test"):
            return base
    return None


# -----------------------------------------------------------------------------
# Assertion insertion
# -----------------------------------------------------------------------------
def insert_assertions_into_test(
    assertion_free_test: str,
    generated_assertions: str,
    indent: str = "    ",
) -> str:
    """
    Insert generated assertion(s) into the test method body before the final
    closing brace. Preserves structure; uses same indent for inserted lines.
    """
    if not generated_assertions.strip():
        return assertion_free_test

    # Find the last '}' that closes the method (simple: last occurrence, or
    # balance braces). We use: insert before the last line that is only '}' or '}\n'.
    lines = assertion_free_test.rstrip().split("\n")
    if not lines:
        return assertion_free_test + "\n" + indent + generated_assertions.strip()

    # Insert before the final closing brace of the method. If the last non-empty
    # line is a single '}', we insert before it.
    insert_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped == "}":
            insert_idx = i
            break

    assertion_lines = [indent + line for line in generated_assertions.strip().split("\n")]
    new_lines = lines[:insert_idx] + [""] + assertion_lines + [""] + lines[insert_idx:]
    return "\n".join(new_lines)


# -----------------------------------------------------------------------------
# Test class wrapping (optional: if test_case is just a method, wrap in a class)
# -----------------------------------------------------------------------------
def wrap_test_method_in_class(
    method_body: str,
    class_name: str = "GeneratedTest",
    package: str = "",
) -> str:
    """Wrap a single @Test method in a minimal JUnit test class."""
    package_line = f"package {package};\n\n" if package else ""
    imports = (
        "import org.junit.Test;\n"
        "import static org.junit.Assert.*;\n\n"
    )
    return (
        f"{package_line}{imports}public class {class_name} {{\n"
        f"{method_body}\n}}\n"
    )


# -----------------------------------------------------------------------------
# Project runner: compile and run (placeholder / Maven implementation)
# -----------------------------------------------------------------------------
class ProjectRunner:
    """
    Compile and run a test against buggy or fixed project directory.
    Override compile_test and run_test for your build system.
    """

    def __init__(
        self,
        timeout_compile: int = 120,
        timeout_run: int = 60,
        use_maven: bool = True,
    ):
        self.timeout_compile = timeout_compile
        self.timeout_run = timeout_run
        self.use_maven = use_maven

    def compile_test(
        self,
        project_path: Path,
        test_source: str,
        test_class_name: str,
        test_file_relative: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Write test source to project and run compile. Returns (success, output).
        Default: write to src/test/java/.../TestClass.java and run mvn test-compile.
        """
        if test_file_relative is None:
            test_file_relative = f"src/test/java/{test_class_name}.java"
        test_path = project_path / test_file_relative
        test_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            test_path.write_text(test_source, encoding="utf-8")
        except Exception as e:
            return False, str(e)

        if self.use_maven:
            cmd = ["mvn", "-q", "test-compile", "-DskipTests"]
            cwd = str(project_path)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_compile,
                    cwd=cwd,
                )
                out = (result.stdout or "") + (result.stderr or "")
                return result.returncode == 0, out
            except subprocess.TimeoutExpired:
                return False, "Compile timeout"
            except Exception as e:
                return False, str(e)
        return True, ""

    def run_test(
        self,
        project_path: Path,
        test_class_name: str,
        test_method_name: str,
    ) -> Tuple[bool, bool, str]:
        """
        Run single test method. Returns (passed, invalid_execution, output).
        passed: True iff BUILD SUCCESS and 0 test failures (parsed from output).
        invalid_execution: True only for timeout, JVM crash, OOM, etc. — not for
        normal assertion failure or assertThrows. Use mvn clean test for isolation.
        """
        if self.use_maven:
            test_spec = f"{test_class_name}#{test_method_name}"
            cmd = ["mvn", "-q", "clean", "test", f"-Dtest={test_spec}", "-DfailIfNoTests=false"]
            cwd = str(project_path)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_run,
                    cwd=cwd,
                )
                out = (result.stdout or "") + (result.stderr or "")
                passed, invalid_execution = _parse_maven_test_result(out)
                # If parser could not infer (no BUILD SUCCESS/FAILURE), fall back to returncode
                if "BUILD SUCCESS" not in out and "BUILD FAILURE" not in out:
                    passed = result.returncode == 0
                return passed, invalid_execution, out
            except subprocess.TimeoutExpired:
                return False, True, "Run timeout"
            except Exception as e:
                return False, True, str(e)
        return False, True, "No runner configured"

    def run_test_with_coverage(
        self,
        project_path: Path,
        test_class_name: str,
        test_method_name: str,
    ) -> Tuple[bool, bool, str, Optional[Dict[str, float]]]:
        """
        Run single test with JaCoCo coverage. Returns (passed, invalid_execution, output, coverage_dict).
        Same pass/invalid logic as run_test; uses mvn clean for isolation.
        """
        if not self.use_maven:
            passed, inv, out = self.run_test(project_path, test_class_name, test_method_name)
            return passed, inv, out, None
        test_spec = f"{test_class_name}#{test_method_name}"
        cmd = [
            "mvn", "-q", "clean",
            "jacoco:prepare-agent",
            "test", f"-Dtest={test_spec}", "-DfailIfNoTests=false",
            "jacoco:report",
        ]
        cwd = str(project_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_run + 30,
                cwd=cwd,
            )
            out = (result.stdout or "") + (result.stderr or "")
            passed, invalid_execution = _parse_maven_test_result(out)
            if "BUILD SUCCESS" not in out and "BUILD FAILURE" not in out:
                passed = result.returncode == 0
            coverage = self._parse_jacoco_report(project_path)
            return passed, invalid_execution, out, coverage
        except subprocess.TimeoutExpired:
            return False, True, "Run timeout", None
        except Exception as e:
            return False, True, str(e), None

    def _parse_jacoco_report(self, project_path: Path) -> Optional[Dict[str, float]]:
        """Parse JaCoCo XML report; return line_pct, method_pct, branch_pct, full_pct or None."""
        report_path = project_path / "target" / "site" / "jacoco" / "jacoco.xml"
        if not report_path.exists():
            return None
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            totals = defaultdict(lambda: {"covered": 0, "missed": 0})
            for counter in root.iter("counter"):
                t = counter.get("type")
                if t:
                    totals[t]["covered"] += int(counter.get("covered", 0))
                    totals[t]["missed"] += int(counter.get("missed", 0))
            out = {}
            for key, name in [("LINE", "line_pct"), ("METHOD", "method_pct"), ("BRANCH", "branch_pct")]:
                c, m = totals[key]["covered"], totals[key]["missed"]
                total = c + m
                out[name] = round(100.0 * c / total, 2) if total else 0.0
            # full_pct = average of line, method, branch (full coverage summary)
            out["full_pct"] = round(
                (out["line_pct"] + out["method_pct"] + out["branch_pct"]) / 3.0, 2
            )
            return out
        except Exception:
            return None


# -----------------------------------------------------------------------------
# Effectiveness evaluation (single instance, single mode)
# -----------------------------------------------------------------------------
def evaluate_single(
    entry: Dict[str, Any],
    mode: str,
    assertions_text: str,
    buggy_path: Optional[Path],
    fixed_path: Optional[Path],
    runner: ProjectRunner,
    test_class_name: Optional[str] = None,
    test_method_name: Optional[str] = None,
    test_file_relative: Optional[str] = None,
    use_isolated_test_class: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one (entry, mode) pair. Returns dict with outcome and execution details.
    """
    result = {
        "mode": mode,
        "outcome": OUTCOME_NOT_EVALUATED,
        "compiles": None,
        "buggy_passed": None,
        "fixed_passed": None,
        "runtime_error": False,
        "message": "",
        "compile_output": None,
        "buggy_run_output": None,
        "fixed_run_output": None,
    }

    assertion_free = get_assertion_free_test(entry)
    if not assertion_free:
        result["outcome"] = OUTCOME_NOT_EVALUATED
        result["message"] = "No assertion-free test body"
        return result

    if not assertions_text or not assertions_text.strip():
        result["outcome"] = OUTCOME_EMPTY_ASSERTION
        result["message"] = "Empty generated assertion"
        return result

    # Unique test class per (bug_id, mode) to avoid overwriting across entries/modes (test isolation)
    base_tcn = test_class_name or get_test_class_name(entry) or "GeneratedTest"
    if use_isolated_test_class:
        bid = get_bug_id(entry)
        tcn = "GenT_" + _sanitize_for_class_name(bid, 10) + "_" + _sanitize_for_class_name(mode, 14)
    else:
        tcn = base_tcn
    tmn = test_method_name or get_test_method_name(entry)
    if not tmn:
        result["outcome"] = OUTCOME_NOT_EVALUATED
        result["message"] = "Could not infer test method name"
        return result

    test_with_assertion = insert_assertions_into_test(assertion_free, assertions_text)
    # If the input is a full class, use it; else wrap
    if "public class " in test_with_assertion and "@Test" in test_with_assertion:
        full_source = test_with_assertion
    else:
        full_source = wrap_test_method_in_class(test_with_assertion, class_name=tcn)

    if not buggy_path or not fixed_path or not buggy_path.is_dir() or not fixed_path.is_dir():
        result["outcome"] = OUTCOME_NOT_EVALUATED
        result["message"] = "Buggy or fixed project path not configured or missing"
        return result

    # Compile on buggy
    comp_ok, comp_out = runner.compile_test(
        buggy_path, full_source, tcn, test_file_relative
    )
    result["compile_output"] = comp_out[:2000] if comp_out else None
    if not comp_ok:
        result["outcome"] = OUTCOME_COMPILE_ERROR
        result["compiles"] = False
        result["message"] = "Compilation failed on buggy project"
        return result
    result["compiles"] = True

    # Run on buggy (we expect failure for an effective assertion)
    buggy_passed, buggy_invalid, buggy_out = runner.run_test(buggy_path, tcn, tmn)
    result["buggy_passed"] = buggy_passed
    result["buggy_run_output"] = buggy_out[:2000] if buggy_out else None
    result["runtime_error"] = buggy_invalid  # True only for invalid execution (timeout, JVM crash)
    # Do NOT return RUNTIME_ERROR here: valid oracles can fail with exceptions (e.g. assertThrows).
    # Only classify as RUNTIME_ERROR when execution was invalid (handled after fixed run).

    # Compile on fixed (same source)
    comp_fixed_ok, _ = runner.compile_test(
        fixed_path, full_source, tcn, test_file_relative
    )
    if not comp_fixed_ok:
        result["outcome"] = OUTCOME_COMPILE_ERROR
        result["message"] = "Compilation failed on fixed project"
        return result

    fixed_passed, fixed_invalid, fixed_out = runner.run_test(fixed_path, tcn, tmn)
    result["fixed_passed"] = fixed_passed
    result["fixed_run_output"] = fixed_out[:2000] if fixed_out else None
    result["runtime_error"] = result["runtime_error"] or fixed_invalid
    # RUNTIME_ERROR only when execution was invalid (timeout, JVM crash, OOM), not assertion failure
    if result["runtime_error"]:
        result["outcome"] = OUTCOME_RUNTIME_ERROR
        result["message"] = "Invalid execution (timeout or JVM crash) on buggy or fixed version"
        return result

    # Classify by pass/fail (semantic oracle effectiveness)
    if not buggy_passed and fixed_passed:
        result["outcome"] = OUTCOME_EFFECTIVE
        result["message"] = "Fails on buggy, passes on fixed"
    elif buggy_passed and fixed_passed:
        result["outcome"] = OUTCOME_INEFFECTIVE_PASS_BOTH
        result["message"] = "Passes on both (does not detect bug)"
    elif not buggy_passed and not fixed_passed:
        result["outcome"] = OUTCOME_INEFFECTIVE_FAIL_BOTH
        result["message"] = "Fails on both"
    else:
        result["outcome"] = OUTCOME_INEFFECTIVE_PASS_BUGGY_FAIL_FIXED
        result["message"] = "Passes on buggy, fails on fixed"

    return result


# -----------------------------------------------------------------------------
# Project path resolution
# -----------------------------------------------------------------------------
def load_project_paths(path: Optional[Path]) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """
    Load mapping from (project_name or bug_id) to (buggy_dir, fixed_dir).
    JSON format: { "project_name": ["/path/buggy", "/path/fixed"], ... }
    or { "bug_id": ["/path/buggy", "/path/fixed"], ... }
    """
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[k.strip()] = (Path(v[0]) if v[0] else None, Path(v[1]) if v[1] else None)
        return out
    except Exception:
        return {}


def resolve_project_paths(
    entry: Dict[str, Any],
    by_project: Dict[str, Tuple[Optional[Path], Optional[Path]]],
    by_bug_id: Optional[Dict[str, Tuple[Optional[Path], Optional[Path]]]] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve (buggy_path, fixed_path) for an entry."""
    by_bug_id = by_bug_id or {}
    bug_id = get_bug_id(entry)
    project = get_project_name(entry)
    if bug_id and bug_id in by_bug_id:
        return by_bug_id[bug_id]
    if project and project in by_project:
        return by_project[project]
    return None, None


# -----------------------------------------------------------------------------
# Commit-based project materialization (no manual project paths)
# -----------------------------------------------------------------------------
def load_repos_config(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load repos config for commit-based materialization.
    JSON format:
      {
        "workdir": "/optional/base/for/clones",
        "repos": {
          "project_name": { "repo_url": "https://...", "branch": "main" },
          ...
        }
      }
    or flat: { "project_name": "https://...", ... } (repo_url only).
    """
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        out = {"workdir": None, "repos": {}}
        if "workdir" in data and data["workdir"]:
            out["workdir"] = Path(data["workdir"])
        repos = data.get("repos", data)
        for k, v in repos.items():
            if k == "workdir":
                continue
            if isinstance(v, str):
                out["repos"][k.strip()] = {"repo_url": v.strip(), "branch": None}
            elif isinstance(v, dict) and v.get("repo_url"):
                out["repos"][k.strip()] = {
                    "repo_url": v["repo_url"].strip(),
                    "branch": v.get("branch") or None,
                }
        return out
    except Exception:
        return {}


def materialize_buggy_fixed_from_commit(
    entry: Dict[str, Any],
    repos_config: Dict[str, Any],
    workdir: Path,
    cache: Dict[Tuple[str, str], Tuple[Path, Path]],
    materialized_dirs: List[Path],
    timeout_clone: int = 300,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Materialize buggy and fixed project dirs from git using entry's bug_id (commit)
    and project_name. Uses parent commit for buggy, commit for fixed.
    Returns (buggy_path, fixed_path) or (None, None) on failure.
    Caches by (project_name, bug_id). Appends created dirs to materialized_dirs for cleanup.
    """
    bug_id = get_bug_id(entry)
    project = get_project_name(entry)
    if not bug_id or not project:
        return None, None
    key = (project, bug_id)
    if key in cache:
        return cache[key]
    repos = repos_config.get("repos", {})
    repo_info = repos.get(project)
    if not repo_info or not repo_info.get("repo_url"):
        logger.debug("No repo_url for project %s; skipping materialization", project)
        return None, None
    repo_url = repo_info["repo_url"]
    branch = repo_info.get("branch")
    short_id = _sanitize_for_class_name(bug_id, 12)
    base = workdir / f"mat_{project}_{short_id}"
    buggy_dir = base.with_name(base.name + "_buggy")
    fixed_dir = base.with_name(base.name + "_fixed")
    try:
        for label, dest, rev in [
            ("buggy", buggy_dir, f"{bug_id}^"),
            ("fixed", fixed_dir, bug_id),
        ]:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            clone_cmd = ["git", "clone", "--quiet", "--no-checkout", repo_url, str(dest)]
            if branch:
                clone_cmd.extend(["--branch", branch])
            r = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=timeout_clone, cwd=str(workdir))
            if r.returncode != 0:
                logger.warning("Clone failed for %s %s: %s", project, label, r.stderr[:500] if r.stderr else r.stdout)
                if buggy_dir.exists():
                    shutil.rmtree(buggy_dir, ignore_errors=True)
                if fixed_dir.exists():
                    shutil.rmtree(fixed_dir, ignore_errors=True)
                return None, None
            # Resolve rev (e.g. bug_id^) to hash so we avoid shell escaping
            rev_parse = subprocess.run(
                ["git", "rev-parse", rev],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(dest),
            )
            if rev_parse.returncode != 0:
                logger.warning("rev-parse %s failed for %s: %s", rev, project, rev_parse.stderr[:300])
                shutil.rmtree(dest, ignore_errors=True)
                return None, None
            checkout_rev = rev_parse.stdout.strip()
            checkout = subprocess.run(
                ["git", "checkout", "--quiet", checkout_rev],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(dest),
            )
            if checkout.returncode != 0:
                logger.warning("Checkout %s failed for %s %s: %s", checkout_rev[:8], project, label, checkout.stderr[:300])
                shutil.rmtree(dest, ignore_errors=True)
                return None, None
            materialized_dirs.append(dest)
        cache[key] = (buggy_dir, fixed_dir)
        logger.info("Materialized %s @ %s (buggy=%s, fixed=%s)", project, bug_id[:8], buggy_dir.name, fixed_dir.name)
        return buggy_dir, fixed_dir
    except subprocess.TimeoutExpired:
        logger.warning("Materialization timeout for %s %s", project, bug_id[:8])
        for d in (buggy_dir, fixed_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        return None, None
    except Exception as e:
        logger.warning("Materialization failed for %s %s: %s", project, bug_id[:8], e)
        for d in (buggy_dir, fixed_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        return None, None


# -----------------------------------------------------------------------------
# Data loading: merged vs dataset + oracles
# -----------------------------------------------------------------------------
def load_merged_input(path: Path) -> List[Dict[str, Any]]:
    """Load a single JSON that already has assertion_free_test_code and generated_assertions_by_mode."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    return [data]


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset JSON (e.g. strengthened_comments_full.json)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_oracle_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load oracle JSON; return dict bug_id -> entry (with generated_oracle.generated_assertions)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else []
    by_bug_id = {}
    for e in entries:
        bid = get_bug_id(e)
        if bid:
            by_bug_id[bid] = e
    return by_bug_id


def merge_dataset_with_oracles(
    dataset: List[Dict[str, Any]],
    oracles_by_mode: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    oracles_by_mode: { mode_name: { bug_id: entry_with_generated_oracle } }.
    Returns list of merged entries; each has assertion_free_test_code and
    generated_assertions_by_mode (only for modes present for that bug_id).
    """
    merged = []
    for entry in dataset:
        bug_id = get_bug_id(entry)
        if not bug_id:
            continue
        assertion_free = get_assertion_free_test(entry)
        if not assertion_free:
            continue
        by_mode = {}
        for mode, oracle_by_bid in oracles_by_mode.items():
            o = oracle_by_bid.get(bug_id)
            if o is None:
                continue
            assertions = (o.get("generated_oracle") or {}).get("generated_assertions", "")
            by_mode[mode] = (assertions or "").strip()
        if not by_mode:
            continue
        new_entry = dict(entry)
        new_entry["assertion_free_test_code"] = assertion_free
        new_entry["generated_assertions_by_mode"] = by_mode
        merged.append(new_entry)
    return merged


# -----------------------------------------------------------------------------
# Batch evaluation and aggregation
# -----------------------------------------------------------------------------
def run_batch(
    entries: List[Dict[str, Any]],
    modes: List[str],
    project_paths_path: Optional[Path],
    runner: ProjectRunner,
    repos_config_path: Optional[Path] = None,
    materialize_workdir: Optional[Path] = None,
    keep_materialized: bool = False,
    timeout_clone: int = 900,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Evaluate all (entry, mode) pairs. Returns (per_instance_results, aggregate_stats).
    When repos_config_path and materialize_workdir are set, materializes buggy/fixed from
    git (commit^ and commit) when project_paths do not provide paths; cleans up unless keep_materialized.
    """
    by_project = load_project_paths(project_paths_path) if project_paths_path else {}
    # project_paths JSON from generate_project_paths_from_local_repos.py is keyed by bug_id; use same dict for lookup
    by_bug_id = by_project
    repos_config = load_repos_config(repos_config_path) if repos_config_path else {}
    materialize_workdir = materialize_workdir or repos_config.get("workdir") or (Path(tempfile.gettempdir()) / "assertion_eval_materialized")
    materialize_workdir = Path(materialize_workdir)
    if repos_config.get("repos"):
        materialize_workdir.mkdir(parents=True, exist_ok=True)
    materialize_cache: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    materialized_dirs: List[Path] = []
    per_instance = []
    # Pre-initialize aggregate schemas so all modes contain all possible outcome keys (even if counts are 0).
    aggregate: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        aggregate[mode] = {"total": 0, "compiles_ok": 0, "effective": 0}
        for out in ALL_OUTCOMES:
            aggregate[mode][out] = 0

    total_entries = len(entries)
    for idx, entry in enumerate(entries):
        logger.info("Entry %d/%d (bug_id=%s, project=%s)", idx + 1, total_entries, get_bug_id(entry)[:8] if get_bug_id(entry) else "?", get_project_name(entry))
        bug_id = get_bug_id(entry)
        buggy_path, fixed_path = resolve_project_paths(entry, by_project, by_bug_id)
        if (buggy_path is None or fixed_path is None) and repos_config.get("repos"):
            buggy_path, fixed_path = materialize_buggy_fixed_from_commit(
                entry, repos_config, materialize_workdir, materialize_cache, materialized_dirs,
                timeout_clone=timeout_clone,
            )
        by_mode = get_generated_assertions_by_mode(entry)
        if not by_mode:
            logger.warning("Entry %s has no generated_assertions_by_mode", bug_id or idx)
            continue

        # When paths are missing (materialization failed or not configured), record NOT_EVALUATED for all modes and continue
        if buggy_path is None or fixed_path is None or not buggy_path.is_dir() or not fixed_path.is_dir():
            skip_reason = "Buggy or fixed project path not configured or materialization failed (clone/checkout timeout or error)"
            logger.warning("Skipping entry %s (%s): %s", bug_id[:8] if bug_id else idx, get_project_name(entry), skip_reason)
            for mode in modes:
                if mode not in by_mode:
                    continue
                aggregate[mode]["total"] += 1
                aggregate[mode][OUTCOME_NOT_EVALUATED] += 1
                per_instance.append({
                    "bug_id": bug_id,
                    "project_name": get_project_name(entry),
                    "mode": mode,
                    "outcome": OUTCOME_NOT_EVALUATED,
                    "compiles": None,
                    "buggy_passed": None,
                    "fixed_passed": None,
                    "message": skip_reason,
                })
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info("Progress: %d/%d entries done (%d instances so far)", idx + 1, total_entries, len(per_instance))
            continue

        for mode in modes:
            if mode not in by_mode:
                continue
            assertions_text = by_mode[mode]
            res = evaluate_single(
                entry,
                mode,
                assertions_text,
                buggy_path,
                fixed_path,
                runner,
            )
            outcome = res["outcome"]
            aggregate[mode]["total"] += 1
            aggregate[mode][outcome] += 1
            if res["compiles"] is True:
                aggregate[mode]["compiles_ok"] += 1
            if outcome == OUTCOME_EFFECTIVE:
                aggregate[mode]["effective"] += 1

            row = {
                "bug_id": bug_id,
                "project_name": get_project_name(entry),
                "mode": mode,
                "outcome": outcome,
                "compiles": res["compiles"],
                "buggy_passed": res["buggy_passed"],
                "fixed_passed": res["fixed_passed"],
                "message": res.get("message", ""),
            }
            per_instance.append(row)
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info("Progress: %d/%d entries done (%d instances so far)", idx + 1, total_entries, len(per_instance))

    # Cleanup materialized dirs unless keep_materialized
    if materialized_dirs and not keep_materialized:
        for d in materialized_dirs:
            if d.exists():
                try:
                    shutil.rmtree(d)
                    logger.debug("Removed materialized dir %s", d)
                except Exception as e:
                    logger.warning("Could not remove %s: %s", d, e)

    return per_instance, dict(aggregate)


def print_aggregate_report(aggregate: Dict[str, Dict[str, Any]], modes: List[str]) -> None:
    """Print summary table, key metrics, and optional coverage averages."""
    print("\n" + "=" * 80)
    print("ASSERTION EFFECTIVENESS EVALUATION — AGGREGATE REPORT")
    print("=" * 80)

    for mode in modes:
        s = aggregate.get(mode, {})
        total = s.get("total", 0)
        if total == 0:
            continue
        compiles_ok = s.get("compiles_ok", 0)
        effective = s.get("effective", 0)
        comp_rate = 100.0 * compiles_ok / total if total else 0
        eff_rate = 100.0 * effective / total if total else 0
        print(f"\n--- {mode} ---")
        print(f"  Total instances:     {total}")
        print(f"  Compilation rate:    {compiles_ok}/{total} ({comp_rate:.1f}%)")
        print(f"  Effectiveness rate:  {effective}/{total} ({eff_rate:.1f}%)")
        print("  Outcome breakdown:")
        for out in ALL_OUTCOMES:
            c = s.get(out, 0)
            if c > 0:
                print(f"    {out}: {c} ({100.0*c/total:.1f}%)")

    print("\n" + "=" * 80)


def write_detailed_json(per_instance: List[Dict[str, Any]], aggregate: Dict, path: Path) -> None:
    """Write detailed results JSON (per-instance + aggregate)."""
    out = {
        "per_instance": per_instance,
        "aggregate": aggregate,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Wrote detailed JSON: %s", path)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------
def parse_oracle_spec(s: str) -> Tuple[str, Path]:
    """Parse 'mode_name:path/to/file.json' -> (mode_name, path)."""
    if ":" in s:
        mode, p = s.split(":", 1)
        return mode.strip(), Path(p.strip())
    return Path(s).stem, Path(s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate assertion effectiveness (compile, fail-on-buggy, pass-on-fixed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Single merged JSON with assertion_free_test_code and generated_assertions_by_mode",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset JSON (e.g. strengthened_comments_full.json); use with --oracles",
    )
    parser.add_argument(
        "--oracles",
        nargs="+",
        metavar="MODE:path.json",
        help="Oracle files: mode1:path1.json mode2:path2.json ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--project-paths",
        type=Path,
        help="JSON mapping project_name or bug_id to [buggy_dir, fixed_dir]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only merge and validate input; do not run compile/test (paths optional)",
    )
    parser.add_argument(
        "--timeout-compile",
        type=int,
        default=120,
        help="Timeout for compile step (seconds)",
    )
    parser.add_argument(
        "--timeout-run",
        type=int,
        default=60,
        help="Timeout for test run (seconds)",
    )
    parser.add_argument(
        "--repos",
        type=Path,
        metavar="repos.json",
        help="JSON config for commit-based materialization: maps project_name to repo_url; optional workdir for clones",
    )
    parser.add_argument(
        "--materialize-workdir",
        type=Path,
        help="Directory for git clones when using --repos (default: repos.workdir or system temp)",
    )
    parser.add_argument(
        "--keep-materialized",
        action="store_true",
        help="Do not delete materialized buggy/fixed clones after evaluation (for debugging)",
    )
    parser.add_argument(
        "--timeout-clone",
        type=int,
        default=900,
        help="Timeout for git clone when using --repos (seconds); default 900 for large repos (e.g. Hadoop)",
    )
    args = parser.parse_args()

    entries: List[Dict[str, Any]] = []
    modes: List[str] = []

    if args.input and args.input.exists():
        entries = load_merged_input(args.input)
        if entries:
            modes = list(set().union(*(get_generated_assertions_by_mode(e).keys() for e in entries)))
        logger.info("Loaded merged input: %d entries, modes: %s", len(entries), modes)
    elif args.dataset and args.dataset.exists() and args.oracles:
        dataset = load_dataset(args.dataset)
        oracles_by_mode = {}
        for spec in args.oracles:
            mode_name, path = parse_oracle_spec(spec)
            if not path.exists():
                logger.warning("Oracle file not found: %s", path)
                continue
            oracles_by_mode[mode_name] = load_oracle_file(path)
        if not oracles_by_mode:
            logger.error("No oracle files loaded")
            return 1
        entries = merge_dataset_with_oracles(dataset, oracles_by_mode)
        modes = list(oracles_by_mode.keys())
        logger.info("Merged dataset + oracles: %d entries, modes: %s", len(entries), modes)
    else:
        if args.input and not args.input.exists():
            logger.error("Input file not found: %s. Create it first with build_evaluation_merged_input.py (see README_EVALUATION.md).", args.input)
        else:
            logger.error("Provide --input (merged JSON) or --dataset + --oracles")
        return 1

    if not entries or not modes:
        logger.error("No entries or modes to evaluate")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = ProjectRunner(
        timeout_compile=args.timeout_compile,
        timeout_run=args.timeout_run,
        use_maven=True,
    )

    if args.dry_run:
        logger.info("Dry run: skipping compile/run. Total entries=%d, modes=%s", len(entries), modes)
        # Write a placeholder result with NOT_EVALUATED for all
        per_instance = []
        for e in entries:
            bid = get_bug_id(e)
            for mode in modes:
                by_mode = get_generated_assertions_by_mode(e)
                if mode not in by_mode:
                    continue
                txt = by_mode[mode]
                outcome = OUTCOME_EMPTY_ASSERTION if not (txt and txt.strip()) else OUTCOME_NOT_EVALUATED
                per_instance.append({
                    "bug_id": bid,
                    "project_name": get_project_name(e),
                    "mode": mode,
                    "outcome": outcome,
                    "compiles": None,
                    "buggy_passed": None,
                    "fixed_passed": None,
                    "message": "Dry run",
                })
        aggregate: Dict[str, Dict[str, Any]] = {}
        for m in modes:
            aggregate[m] = {"total": 0, "compiles_ok": 0, "effective": 0}
            for out in ALL_OUTCOMES:
                aggregate[m][out] = 0
        for r in per_instance:
            m = r["mode"]
            aggregate[m]["total"] += 1
            aggregate[m][r["outcome"]] += 1
        print_aggregate_report(dict(aggregate), modes)
        write_detailed_json(per_instance, dict(aggregate), args.output_dir / "effectiveness_detailed.json")
        return 0

    per_instance, aggregate = run_batch(
        entries,
        modes,
        args.project_paths,
        runner,
        repos_config_path=args.repos,
        materialize_workdir=getattr(args, "materialize_workdir", None),
        keep_materialized=getattr(args, "keep_materialized", False),
        timeout_clone=getattr(args, "timeout_clone", 900),
    )
    print_aggregate_report(aggregate, modes)
    write_detailed_json(per_instance, aggregate, args.output_dir / "effectiveness_detailed.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
