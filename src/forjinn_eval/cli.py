"""``forjinn-eval`` console command.

Subcommands
-----------
run    - run a Python *module* (or ``-c`` expression) that defines
         ``@forjinn_eval.agent_test`` cases, then print a report and write
         JSON/JUnit/Markdown.
list   - list registered agent_test cases from a module.
smoke  - offline smoke test against the bundled recorded samples (no network).

Examples
--------
    forjinn-eval run my_agent_tests.py            # run all cases in the file
    forjinn-eval run my_agent_tests.py -k count   # only cases whose name has 'count'
    forjinn-eval run my_agent_tests.py --min-pass-rate 0.9 --fail-on-gate
    forjinn-eval smoke --report report.json
    forjinn-eval list my_agent_tests.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location(_mod_name(path), path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mod_name(path: str) -> str:
    return Path(path).stem


def cmd_run(args) -> int:
    from forjinn_eval import SuiteRunner, build_cases, registered_cases

    mod = _load_module(args.module)
    _ = mod  # registration happened on import
    names = registered_cases()
    if args.k:
        names = [n for n in names if all(k in n for k in args.k)]
    if not names:
        print("no matching registered agent_test cases")
        return 1
    print(f"Running {len(names)} case(s): {names}")
    cases = build_cases(names)
    sr = SuiteRunner(parallel=args.parallel, suite_name=args.suite_name).run(cases)

    print()
    print(sr.to_markdown())

    for label, path, content in (
        ("JSON", args.report_json, json.dumps(sr.to_dict(), indent=2)),
        ("JUnit", args.report_xml, sr.to_junit_xml()),
        ("Markdown", args.report_md, sr.to_markdown()),
    ):
        if not path:
            continue
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"wrote {label}: {p}")

    # exit code
    if args.fail_on_gate:
        return 0 if (args.min_pass_rate is None or (sr.pass_rate or 0) >= args.min_pass_rate) else 2
    return 0 if (sr.failed == 0 and sr.errors == 0) else 2


def cmd_list(args) -> int:
    _load_module(args.module)
    from forjinn_eval import registered_cases

    for n in registered_cases():
        print(n)
    return 0


def cmd_smoke(args) -> int:
    from forjinn_eval import (
        AllNodesFinished,
        AgentRun,
        OutputMatchesRegex,
        OutputNotEmpty,
        NoCostLeakage,
        SuiteRunner,
        make_case,
    )

    here = Path(__file__).resolve().parent
    samples_dir = here.parent.parent / "tests" / "fixtures"
    if not samples_dir.is_dir():
        samples_dir = here.parent.parent.parent / "research" / "captured"

    a1 = samples_dir / "agent1_count.json"
    a2 = samples_dir / "agent2_tools_nonstream.json"

    cases = []
    if a1.is_file():
        run1 = AgentRun.from_nonstream("agent-1", json.loads(a1.read_text(encoding="utf-8")))
        cases.append(make_case(
            "smoke-agent-1-count",
            run=run1,
            evaluators=[
                AllNodesFinished(),
                OutputNotEmpty(),
                NoCostLeakage(),
                OutputMatchesRegex(r"(?m)^5$"),
            ],
            tags=["smoke", "agent-1"],
        ))
    if a2.is_file():
        run2 = AgentRun.from_nonstream("agent-2", json.loads(a2.read_text(encoding="utf-8")))
        cases.append(make_case(
            "smoke-agent-2-mcp",
            run=run2,
            evaluators=[AllNodesFinished(), OutputNotEmpty()],
            tags=["smoke", "agent-2"],
        ))
    if not cases:
        print("no recorded samples found (looked in tests/fixtures and research/captured)")
        return 1
    sr = SuiteRunner(parallel=1, suite_name="forjinn-eval-smoke").run(cases)
    print(sr.to_markdown())
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sr.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote JSON: {p}")
    return 0 if (sr.failed == 0 and sr.errors == 0) else 2


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="forjinn-eval", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run agent_test cases from a module file")
    pr.add_argument("module")
    pr.add_argument("-k", action="append", default=[], help="filter by name substring (repeatable)")
    pr.add_argument("--parallel", type=int, default=1)
    pr.add_argument("--suite-name", default="forjinn-eval")
    pr.add_argument("--report-json")
    pr.add_argument("--report-xml")
    pr.add_argument("--report-md")
    pr.add_argument("--min-pass-rate", type=float, default=None)
    pr.add_argument("--fail-on-gate", action="store_true", help="exit 2 if min-pass-rate not met")
    pr.set_defaults(func=cmd_run)

    pl = sub.add_parser("list", help="list registered agent_test cases in a module")
    pl.add_argument("module")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("smoke", help="offline smoke against recorded samples")
    ps.add_argument("--report", help="write JSON report to this path")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
