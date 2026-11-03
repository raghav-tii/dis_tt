"""Framework-agnostic orchestrator for the dis_tt tasks.

You plug in ONE function, ``run_agent`` (bottom of the "HOOK" section). Everything else
(task graph, prompt composition, git worktrees, verification gates, retries, merging,
reports) is handled here, using only git, pytest and the standard library.

Usage
-----
    python agents/orchestrate.py --dry-run T1          # print the exact prompts for T1
    python agents/orchestrate.py T1 T2 T3              # run those tasks (deps must be merged)
    python agents/orchestrate.py --all                 # run the whole DAG in dependency waves
    python agents/orchestrate.py --all --parallel 3    # up to 3 agents at once within a wave

Gates (the orchestrator NEVER trusts an agent's report)
-------------------------------------------------------
1. Protected files unchanged: every test file that existed at baseline, and problems/toy.py.
2. The task's acceptance command passes.
3. No regression: every test passing on main before the task still passes on the branch.
4. The agent's final message contains a parseable report with STATUS: done.
If gate 1 fails, or the report says blocked or has a non-empty ISSUES, the task stops for
human review. Gates 2-4 failing trigger a retry with the gate log as feedback
(up to --max-attempts).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "agents" / "skills" / "dis-tt-implementer" / "SKILL.md"
AGENTS_MD = REPO / "AGENTS.md"
WORKTREES = REPO.parent / f"{REPO.name}_worktrees"
REPORTS = REPO / "agents" / "reports"

PROTECTED = ["tests/conftest.py", "tests/test_toy_problem.py", "src/dis_tt/problems/toy.py"]


@dataclass
class Task:
    deps: list[str]
    accept: str | None          # pytest target(s); None -> suite + report only (human review)
    owns: list[str]
    extra_accept_glob: str | None = None  # agent-written tests that must exist and pass
    optional: bool = False


TASKS: dict[str, Task] = {
    "T1": Task([], "tests/test_tt.py", ["src/dis_tt/tt.py"]),
    "T2": Task([], "tests/test_linalg.py", ["src/dis_tt/linalg.py"]),
    "T3": Task([], "tests/test_snapshots.py -m 'slow or not slow'", ["src/dis_tt/snapshots.py"]),
    "T4": Task(["T1", "T2"], "tests/test_dense.py", ["src/dis_tt/dense.py"]),
    "T5": Task(["T1", "T4"], "tests/test_native.py", ["src/dis_tt/native.py"]),
    "T6": Task(["T1", "T3", "T4"], "tests/test_input_side.py", ["src/dis_tt/input_side.py"]),
    "T6b": Task(["T1", "T4"], "tests/test_bounds.py", ["src/dis_tt/bounds.py"]),
    "T7": Task(["T3", "T6"], "tests/test_rbno.py", ["src/dis_tt/rbno.py"]),
    "T8": Task(["T7"], None, ["src/dis_tt/rbno.py"], extra_accept_glob="tests/test_train*.py"),
    "T9": Task(["T5", "T6b", "T7"], None, ["experiments/", "src/dis_tt/problems/"]),
    "T10": Task(["T4", "T6b"], None, ["src/dis_tt/refine.py"], extra_accept_glob="tests/test_refine*.py", optional=True),
}


# ======================================================================= HOOK
def run_agent(system_prompt: str, user_prompt: str, workdir: Path, task_id: str, attempt: int) -> str:
    """Run ONE agent session to completion and return its final message text.

    Contract for your framework:
      * The agent's working directory (and file-tool root) is ``workdir``: a git
        worktree on branch ``task/<id>``. It must not touch anything outside it.
      * Tools the agent needs: read/write/edit files, run shell commands (python,
        pytest). It does NOT need git; the orchestrator commits.
      * ``system_prompt`` = the skill + AGENTS.md. Use it as the system prompt, or load
        agents/skills/ natively if your framework supports SKILL.md skills (then keep
        only the AGENTS.md part here).
      * ``user_prompt`` = the task prompt (+ gate feedback on retries).
      * Return the agent's FINAL assistant message (must contain the report block).
      * Budget suggestion: 60-150 tool calls for T1-T7; more for T9.

    Example skeleton (subprocess to your own CLI):
        out = subprocess.run(["my-agent", "--cwd", str(workdir),
                              "--system", system_prompt, "--prompt", user_prompt],
                             capture_output=True, text=True, timeout=3 * 3600)
        return out.stdout
    """
    raise NotImplementedError("Plug your agent framework into agents/orchestrate.py::run_agent")
# ======================================================================= /HOOK


def sh(cmd: str, cwd: Path, check: bool = False, timeout: int = 3600) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"`{cmd}` failed in {cwd}:\n{p.stdout}\n{p.stderr}")
    return p


# ------------------------------------------------------------------ prompts
def load_task_prompts() -> dict[str, str]:
    """Parse the '### Tn: title' + '> ...' blocks of AGENTS.md."""
    text = AGENTS_MD.read_text()
    out = {}
    for m in re.finditer(r"^### (T\w+): [^\n]*\n((?:>.*\n?)+)", text, flags=re.M):
        out[m.group(1)] = "\n".join(line[1:].strip() for line in m.group(2).splitlines())
    return out


def compose(task_id: str, workdir: Path, feedback: str | None) -> tuple[str, str]:
    skill = SKILL.read_text()
    agents_md = AGENTS_MD.read_text()
    system = (f"{skill}\n\n---\n# Repository instructions (AGENTS.md)\n\n{agents_md}\n")
    t = TASKS[task_id]
    user = (
        f"Task {task_id}. Your working directory is {workdir} (a git checkout of the dis_tt repo; "
        f"dependencies {', '.join(t.deps) or 'none'} are already merged).\n\n"
        f"{load_task_prompts()[task_id]}\n\n"
        f"Files you own: {', '.join(t.owns)}; plus notes/{task_id}.md and new test files. "
        f"Do NOT pip install the package: pytest imports ./src via pyproject (pythonpath); for scripts use PYTHONPATH=src. "
        f"End your final message with the report block from AGENTS.md, starting with 'TASK: {task_id}'."
    )
    if feedback:
        user += ("\n\nPREVIOUS ATTEMPT FAILED THE ORCHESTRATOR'S GATES. Your earlier changes are still in "
                 "the working directory. Fix the problems below; do not weaken or edit protected tests.\n\n"
                 f"{feedback}")
    return system, user


# ------------------------------------------------------------------ git
def ensure_repo() -> None:
    if not (REPO / ".git").exists():
        sh("git init -q -b main && git add -A && git commit -qm 'scaffold'", REPO, check=True)
    WORKTREES.mkdir(exist_ok=True)


def make_worktree(task_id: str) -> Path:
    wt = WORKTREES / task_id
    if wt.exists():
        sh(f"git worktree remove --force {wt}", REPO)
    sh(f"git branch -D task/{task_id}", REPO)
    sh(f"git worktree add -q -b task/{task_id} {wt} main", REPO, check=True)
    return wt


def commit_all(wt: Path, msg: str) -> None:
    sh("git add -A", wt, check=True)
    sh(f"git commit -qm {json.dumps(msg)} --allow-empty", wt, check=True)


# ------------------------------------------------------------------ gates
def passing_tests(cwd: Path) -> set[str]:
    p = sh("python -m pytest -q -rA --tb=no -p no:cacheprovider -m 'slow or not slow'", cwd, timeout=7200)
    return {line.split(" ", 1)[1].strip() for line in p.stdout.splitlines() if line.startswith("PASSED ")}


def protected_violations(wt: Path) -> list[str]:
    base_tests = sh("git ls-tree -r --name-only main tests", wt).stdout.split()
    changed = sh("git diff --name-status main...HEAD", wt).stdout.splitlines()
    bad = []
    for line in changed:
        status, *paths = line.split("\t")
        for path in paths:
            if status[0] in "MDR" and (path in PROTECTED or path in base_tests):
                bad.append(f"{status} {path}")
    return bad


REPORT_RE = re.compile(r"TASK:\s*(\S+).*?STATUS:\s*(\w+).*?ISSUES:\s*(.*?)\n\s*NEXT:", re.S)


def parse_report(msg: str) -> dict | None:
    m = REPORT_RE.search(msg or "")
    if not m:
        return None
    issues = m.group(3).strip()
    return {"task": m.group(1), "status": m.group(2).lower(),
            "issues": "" if issues.lower().startswith("none") else issues}


def gate(task_id: str, wt: Path, baseline: set[str], final_msg: str) -> tuple[str, str]:
    """Return (verdict, log). verdict in {"pass", "retry", "human"}."""
    t, log = TASKS[task_id], []
    bad = protected_violations(wt)
    if bad:
        return "human", "Protected files modified:\n" + "\n".join(bad)
    rep = parse_report(final_msg)
    if rep and (rep["status"] == "blocked" or rep["issues"]):
        return "human", f"Agent reported status={rep['status']} issues={rep['issues']!r}"
    ok = True
    targets = [t.accept] if t.accept else []
    if t.extra_accept_glob:
        found = sorted(str(p.relative_to(wt)) for p in wt.glob(t.extra_accept_glob))
        if not found:
            ok = False
            log.append(f"Missing required agent-written tests matching {t.extra_accept_glob}")
        targets += found
    for target in targets:
        p = sh(f"python -m pytest -q {target}", wt, timeout=7200)
        tail = "\n".join(p.stdout.strip().splitlines()[-25:])
        log.append(f"$ pytest -q {target}  (exit {p.returncode})\n{tail}")
        ok &= p.returncode == 0
    lost = baseline - passing_tests(wt)
    if lost:
        ok = False
        log.append("REGRESSIONS (passed on main, fail now):\n" + "\n".join(sorted(lost)))
    if rep is None:
        ok = False
        log.append("No parseable report block (TASK/STATUS/.../ISSUES/NEXT) in final message.")
    elif rep["status"] != "done":
        ok = False
        log.append(f"Report STATUS is {rep['status']!r}, expected 'done'.")
    return ("pass" if ok else "retry"), "\n\n".join(log)


# ------------------------------------------------------------------ driver
def run_task(task_id: str, max_attempts: int) -> tuple[str, str]:
    wt = make_worktree(task_id)
    baseline = passing_tests(wt)
    feedback, verdict, log = None, "retry", ""
    for attempt in range(1, max_attempts + 1):
        system, user = compose(task_id, wt, feedback)
        t0 = time.time()
        final = run_agent(system, user, wt, task_id, attempt)
        commit_all(wt, f"{task_id} attempt {attempt}")
        verdict, log = gate(task_id, wt, baseline, final)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{task_id}_attempt{attempt}.md").write_text(
            f"# {task_id} attempt {attempt} ({time.time() - t0:.0f}s) -> {verdict}\n\n"
            f"## Agent final message\n\n{final}\n\n## Gate log\n\n```\n{log}\n```\n")
        print(f"[{task_id}] attempt {attempt}: {verdict}", flush=True)
        if verdict != "retry":
            break
        feedback = log
    return verdict, log


def merge(task_id: str) -> None:
    sh(f"git merge -q --no-ff task/{task_id} -m 'merge {task_id}'", REPO, check=True)


def waves(selected: list[str]) -> list[list[str]]:
    done, left, out = set(), list(selected), []
    while left:
        ready = [t for t in left if all(d in done or d not in selected for d in TASKS[t].deps)]
        if not ready:
            raise SystemExit(f"Dependency cycle or missing deps among {left}")
        out.append(ready)
        done |= set(ready)
        left = [t for t in left if t not in ready]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--all", action="store_true", help="all non-optional tasks")
    ap.add_argument("--dry-run", action="store_true", help="print prompts only")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    selected = [t for t in TASKS if not TASKS[t].optional] if args.all else args.tasks
    unknown = [t for t in selected if t not in TASKS]
    if unknown or not selected:
        raise SystemExit(f"Choose tasks from {list(TASKS)} (got {selected})")

    if args.dry_run:
        for t in selected:
            system, user = compose(t, WORKTREES / t, None)
            print(f"===== {t} SYSTEM ({len(system)} chars) =====\n{system[:600]}\n[...]\n"
                  f"===== {t} USER =====\n{user}\n")
        return

    ensure_repo()
    for wave in waves(selected):
        print(f"wave: {wave}", flush=True)
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            results = dict(zip(wave, ex.map(lambda t: run_task(t, args.max_attempts), wave)))
        for t in wave:  # merge sequentially, in DAG order
            verdict, log = results[t]
            if verdict == "pass":
                merge(t)
                print(f"[{t}] merged into main", flush=True)
            else:
                print(f"[{t}] STOPPED ({verdict}). See agents/reports/{t}_attempt*.md\n{log[:2000]}", flush=True)
                raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
