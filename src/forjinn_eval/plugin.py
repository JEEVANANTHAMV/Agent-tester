"""Pytest plugin for forjinn_eval.

Two usage styles:

1. **Inline** - ordinary pytest functions that drive the library::

       def test_agent_1_count():
           from forjinn_eval import ForjinnClient, OutputMatchesRegex, make_case, SuiteRunner
           client = ForjinnClient("https://172.16.34.7")
           run = client.predict("03d5abc5-...", "Count 1..5")
           case = make_case("t", run, [OutputMatchesRegex(r"(?m)^1\\n2\\n3\\n4\\n5$")])
           sr = SuiteRunner(suite_name="x").run([case])
           assert sr.failed == 0 and sr.errors == 0

2. **Registered cases** via ``@forjinn_eval.agent_test``. Run with::

       pytest --forjinn-cases=my_tests.py            # all registered cases
       pytest --forjinn-cases=my_tests.py=count,other
       pytest --forjinn-cases=my_tests.py --forjinn-report=report.json --forjinn-junit=report.xml

   Each registered case runs through :class:`forjinn_eval.SuiteRunner` after
   collection and is reported in the terminal summary as a per-case line that
   is added to pytest's ``passed`` / ``failed`` counters, so the exit code and
   the final total are accurate. ``--forjinn-report`` / ``--forjinn-junit``
   also write a cumulative forjinn-eval report.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def pytest_addoption(parser):
    group = parser.getgroup("forjinn-eval")
    group.addoption(
        "--forjinn-cases",
        action="append", default=[],
        metavar="MODULE[=NAME,...]",
        help="Run @forjinn_eval.agent_test cases from a module/file. "
        "Optional NAME,... filters which cases to run.",
    )
    group.addoption(
        "--forjinn-report", action="store", default=None, metavar="PATH.json",
        help="Write the cumulative forjinn-eval suite result to this JSON path.",
    )
    group.addoption(
        "--forjinn-junit", action="store", default=None, metavar="PATH.xml",
        help="Write forjinn-eval JUnit XML to this path (in addition to pytest's).",
    )


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location(
        "__forjinn_eval__" + Path(path).stem, path
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_spec(spec: str) -> Tuple[str, Optional[List[str]]]:
    if "=" in spec:
        module, names = spec.split("=", 1)
        return module, [n for n in names.split(",") if n]
    return spec, None


def _collect_cases(config) -> list:
    specs: List[str] = config.getoption("--forjinn-cases", [])
    if not specs:
        return []
    from forjinn_eval import build_cases

    collected = []
    for spec in specs:
        module, names = _parse_spec(spec)
        mod = _load_module(module)
        if mod is None:
            raise SystemExit(f"forjinn-eval: could not load cases module: {module}")
        collected.extend(build_cases(names))
    return collected


def pytest_configure(config):
    config._forjinn_eval_cases: list = _collect_cases(config)
    config._forjinn_eval_result: Optional[object] = None


def pytest_collection_finish(session):
    """Run registered forjinn-eval cases right after collection.

    At this point ``session.config`` is finalised and the terminal reporter
    exists, so we can add our outcomes to pytest's ``passed`` / ``failed``
    counters and write the cumulative reports.
    """
    config = session.config
    cases: list = config._forjinn_eval_cases
    if not cases:
        return
    from forjinn_eval import SuiteRunner
    from forjinn_eval.results import Status

    suite_result = SuiteRunner(suite_name="forjinn-eval").run(cases)
    config._forjinn_eval_result = suite_result

    tr = config.pluginmanager.getplugin("terminalreporter")
    tw = getattr(tr, "_tw", None) if tr is not None else None
    if tw is not None:
        tw.write("\nforjinn-eval registered cases\n")
        tw.write("-" * 40 + "\n")

    for r in suite_result.results:
        if r.status is Status.PASS:
            if tr is not None:
                tr._tw.write(f"  forjinn_case[{r.name}] PASSED\n")
        else:
            if tr is not None:
                tr._tw.write(f"  forjinn_case[{r.name}] FAILED\n")
                for c in r.check_results:
                    if c.status in (Status.FAIL, Status.ERROR):
                        tr._tw.write(f"      - {c.name}: {c.reason}\n")
            # Mark the session as failed so the exit code is non-zero.
            session.testsfailed += 1

    # Cumulative reports
    report_json = config.getoption("--forjinn-report", None)
    junit_xml = config.getoption("--forjinn-junit", None)
    if report_json:
        p = Path(report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(suite_result.to_dict(), indent=2), encoding="utf-8")
    if junit_xml:
        p = Path(junit_xml)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(suite_result.to_junit_xml(), encoding="utf-8")
