# Phase 2 Plan — Eval Harness: Production Intelligence + Agent Evaluation

> **Created:** June 22 2026
> **Revised:** June 22 2026 (post-audit)
> **Profile:** eval-harness (build)
> **Status:** Research complete. Audit complete. Ready for execution.
> **Research:** `research/eval-harness/PHASE_2_RESEARCH.md` + `PHASE_2_RESEARCH_SUBAGENT.md`
> **Audit:** `research/eval-harness/PHASE_2_AUDIT.md`

---

## Context

Eval Harness v0.1.1 is shipped: 96 tests, 93% coverage, CI green, PyPI published. Phase 2 was never started.

**Strategy:** Phase 2A = Production Intelligence (deepen core). Phase 2B = Agent Evaluation (environment adapters).

---

## Prerequisites (Before Phase 2A)

1. Audit and close/migrate ONI-35 through ONI-42 (v1.1 items)
2. Establish judge calibration baseline (5 canonical examples through each judge)
3. Run 1-2 day environment adapter feasibility spike

---

## Phase 2A: Production Intelligence (Weeks 1-4)

### Features

| # | Feature | Priority | Effort | Deliverable |
|---|---------|----------|--------|-------------|
| 1 | Trend tracking (`trend`) | P0 | 3-4 days | Score timeline + regression detection |
| 2 | Judge calibration (`calibrate`) | P1 | 2-3 days | Inter-judge agreement scores |
| 3 | Rubric templates (`rubric`) | P1 | 4-5 days | 1→5 built-in templates + custom YAML |
| 4 | CI/CD gate (`gate`) | P1 | 2-3 days | `--baseline` + `--suggest-baseline` |
| 5 | Feedback loop (`--feedback`) | P1 | 3 days | Improvement suggestions for low scores |
| 6 | Multi-judge comparison | P2 | 5 days | `--judge-comparison` with calibration |
| **Total** | | **18-23 days** | **v0.2.0** |

### Edge Cases (ALL features)

- **Sparse data:** `trend` requires ≥3 runs for display, ≥5 for regression
- **Judge unavailable:** `--degrade` flag for local heuristic fallback
- **Invalid input:** Structured field-level errors, not tracebacks
- **Concurrent writes:** File-based lock or `BEGIN IMMEDIATE` retry
- **DB migration failures:** Rollback scripts for every migration, tested
- **No baseline:** `--suggest-baseline` or clear error with guidance

---

## Validation Checkpoint (Between 2A and 2B)

1. **User Validation Sprint (3-5 days):** 3 external users, README-only, measure completion + clarity
2. **Positioning Review:** POSITIONING.md explaining dual-mode vs cua
3. **Environment Feasibility Spike (1-2 days):** Validate python-repl + subprocess agent

---

## Phase 2B: Agent Evaluation (Weeks 5-12)

### Prerequisite: Agent Interface (P0, 2 days)

```python
class Agent(ABC):
    async def start(self) -> None: ...
    async def act(self, observation: str) -> str: ...
    async def stop(self) -> None: ...

class SubprocessAgent(Agent): ...  # CLI subprocess, stdin/stdout
class PythonAgent(Agent): ...     # Import + call Python function
```

### Features

| # | Feature | Priority | Effort | Deliverable |
|---|---------|----------|--------|-------------|
| 7 | Agent eval core | P0 | 3-4 weeks | `eval-harness agent eval` |
| 8 | Task suites | P1 | 1-1.5 weeks | 3 suites (15 tasks) |
| 9 | Env adapters | P1 | 1 week | python-repl + mock (bash-sandbox → Phase 2C) |
| **Total** | | **6-8 weeks** | **v0.3.0** |

### Edge Cases

- **Agent hangs:** `--step-timeout` (30s default), `--trajectory-timeout` (300s default)
- **Trajectory failure:** Incremental DB writes, recoverable up to last step
- **Agent crash:** Recorded as task failure with reason

---

## Database Migrations

### Migration v2 (Phase 2A)
```sql
CREATE TABLE rubric_templates (
    template_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    yaml_content TEXT NOT NULL, is_builtin INTEGER DEFAULT 0, created_at TEXT NOT NULL
);
ALTER TABLE eval_runs ADD COLUMN rubric_template_id TEXT;
```

### Migration v3 (Phase 2B)
```sql
CREATE TABLE agent_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, agent_path TEXT NOT NULL,
    task_suite TEXT NOT NULL, rubric_id TEXT, status TEXT DEFAULT 'running',
    completed_at TEXT, mean_score REAL, pass_rate REAL
);
CREATE TABLE agent_trajectories (
    trajectory_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
    task_name TEXT NOT NULL, success INTEGER, step_count INTEGER, total_reward REAL
);
CREATE TABLE agent_steps (
    step_id TEXT PRIMARY KEY, trajectory_id TEXT NOT NULL REFERENCES agent_trajectories(trajectory_id),
    step_number INTEGER NOT NULL, observation TEXT, action TEXT, reward REAL
);
```

---

## Linear Issues

### Pre-Phase-2A
| Issue | Title | Priority |
|-------|-------|----------|
| ONI-43 | Audit/close ONI-35..ONI-42 (v1.1 cleanup) | P0 |
| ONI-44 | Establish judge calibration baseline | P0 |

### Phase 2A
| Issue | Title | Priority |
|-------|-------|----------|
| ONI-45 | `eval-harness trend` — timeline + regression | P0 |
| ONI-46 | `eval-harness calibrate` — judge calibration | P1 |
| ONI-47 | `eval-harness rubric` — templates + custom YAML | P1 |
| ONI-48 | `eval-harness gate` — CI/CD gate + suggest-baseline | P1 |
| ONI-49 | `--feedback` flag — improvement suggestions | P1 |
| ONI-50 | Multi-judge comparison with calibration | P2 |
| ONI-51 | DB migration v2 (rubric_templates + seed) | P1 |
| ONI-52 | Edge case handling for all new commands | P1 |

### Validation
| Issue | Title | Priority |
|-------|-------|----------|
| ONI-53 | User validation sprint (3 external users) | P0 |
| ONI-54 | Write POSITIONING.md | P1 |
| ONI-55 | Environment feasibility spike | P0 |

### Phase 2B
| Issue | Title | Priority |
|-------|-------|----------|
| ONI-56 | Agent interface (Agent ABC + SubprocessAgent) | P0 |
| ONI-57 | Agent eval core — trajectory + scoring | P0 |
| ONI-58 | Task suites (3 suites, 15 tasks) | P1 |
| ONI-59 | Environment adapters (python-repl + mock) | P1 |
| ONI-60 | DB migration v3 (agent tables) | P0 |

---

## Success Criteria

### Phase 2A (v0.2.0)
- [ ] `eval-harness trend` — score timeline (≥3 runs display, ≥5 regression)
- [ ] `eval-harness calibrate` — inter-judge agreement scores
- [ ] `eval-harness rubric --list` — 1→5 built-in templates
- [ ] `eval-harness gate --baseline 0.8` — exit codes 0/1/2
- [ ] `eval-harness gate --suggest-baseline` — suggests from history
- [ ] `eval-harness run --feedback` — improvement suggestions
- [ ] Edge cases handled (sparse data, judge unavail, invalid input)
- [ ] 110+ tests, 90%+ coverage, CI green, CHANGELOG updated

### Validation
- [ ] 3 external users complete first eval (README-only)
- [ ] POSITIONING.md written
- [ ] Environment feasibility spike complete

### Phase 2B (v0.3.0)
- [ ] `Agent` ABC + `SubprocessAgent` defined and tested
- [ ] `eval-harness agent eval` works end-to-end
- [ ] 3 task suites (15 tasks)
- [ ] python-repl environment adapter working
- [ ] Trajectory recording + scoring functional
- [ ] Agent hang/timeout handling
- [ ] 140+ tests, 90%+ coverage

---

## Execution Rules

1. TDD: failing test before implementation
2. One feature per commit
3. Phase gate after 2A before starting 2B
4. Factory Droid for code generation per feature
5. Update this plan as work progresses

---

*Plan created June 22 2026. Research + audit complete. Ready for build agent execution.*
