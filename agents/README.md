# Running the dis_tt tasks with your own agent framework

## Pieces

| File | Role |
| --- | --- |
| `agents/skills/dis-tt-implementer/SKILL.md` | Procedure every implementing agent follows (orient, TDD loop, DIS-TT debug checklist, guardrails, report format) |
| `AGENTS.md` | Repo rules, conventions, task table, one prompt block per task |
| `agents/orchestrate.py` | Task DAG, prompt composition, git worktrees, verification gates, retries, merging |
| `agents/reports/` | Final message + gate log per attempt (git-ignored) |

## 1. Wire in your framework (one function)

Implement `run_agent(system_prompt, user_prompt, workdir, task_id, attempt) -> str` in
`agents/orchestrate.py`. Your agent needs file read/write/edit and a shell (python, pytest)
rooted at `workdir`. It does not need git. Return its final message.

How to supply the skill:
- **Framework supports SKILL.md skills** (folder with YAML frontmatter, loaded on demand):
  point its skills path at `agents/skills/`. You can then drop the skill text from
  `system_prompt` in `compose()` and keep only the AGENTS.md part.
- **Otherwise:** use `system_prompt` as-is. `compose()` already puts SKILL.md + AGENTS.md into it
  (~20 KB).

## 2. Check the prompts, then run

```
python agents/orchestrate.py --dry-run T1          # inspect exactly what the agent receives
python agents/orchestrate.py T1 T2 T3 --parallel 3 # first wave
python agents/orchestrate.py T4                    # then the rest, wave by wave
python agents/orchestrate.py --all --parallel 3    # or the whole DAG (T10 excluded; run it explicitly)
```

Run wave by wave the first time and read each `agents/reports/<task>_attempt*.md` before
continuing. T4's numbers (per-bond ranks and tails) are the first real check of the method.

## 3. What the gates guarantee

The orchestrator does not trust agent reports. A task merges into `main` only if:
1. no protected or pre-existing test file was modified;
2. its acceptance tests pass (plus agent-written tests for T8 and T10);
3. nothing that passed on `main` before the task now fails;
4. the final message contains a report with `STATUS: done` and `ISSUES: none`.

Gates 2-4 failing produce a retry with the gate log as feedback (`--max-attempts`, default 3).
A protected-file edit, `STATUS: blocked`, or a non-empty `ISSUES` stops the run for you.
That is deliberate: a claimed test bug or theory contradiction needs a human decision.

## 4. Tasks that need your judgement

- **T9** (experiments, PDE problem) has no automatic acceptance beyond "suite still green".
  Review the numbers in `results/` and the report yourself.
- **T6's wide-factor investigation** and any **ISSUE** entries in `notes/` are research questions,
  not bugs.
- **T10** is optional; run it with `python agents/orchestrate.py T10`.

## 5. Without the orchestrator

Per task: create a branch, start an agent with SKILL.md + AGENTS.md as the system prompt, and
send the task's prompt block from AGENTS.md as the first user message. Then run its
acceptance command and `pytest -q -m "slow or not slow"` yourself before merging. Respect
the order: {T1, T2, T3} -> T4 -> {T5, T6, T6b} -> T7 -> T8 -> T9 -> T10.
