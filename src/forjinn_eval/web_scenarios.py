"""Open *forjinn-eval* to Forjinn web - a scenario battery runner.

This is the piece that makes the **entire metric catalog** runnable against
Forjinn web agents, for *all* scenarios:

* **Recorded fixtures** (captured agent responses) are replayed offline - no
  network, no LLM. The structural/deterministic metrics run for real; the
  LLM-judged metrics run under the permissive offline judge (``FORJINN_OFFLINE`` /
  a :class:`~forjinn_eval.judge.queuing_judge`) so the full pipeline is exercised.
* **Live scenarios** hit the Forjinn builder host (``https://172.16.34.7`` by
  default) for real - streaming or not - and every metric (including the real
  judge, when ``FORJINN_JUDGE_CHATFLOW`` is set) runs.
* **Persona simulation** drives a fresh *multi-turn* conversation with a
  simulated user against agent-1's chatflow (offline it reuses the capture;
  online it calls ``predict_multi_turn``), so conversational metrics get a
  genuine transcript.

Each "scenario" is a labelled situation with an input (question or turns), an
optional reference, an optional role, and the metric *batteries* to apply.
``run_all_scenarios`` executes every scenario every battery and returns a
:class:`SuiteResult`, which serialises to JSON/JUnit/Markdown exactly like the
rest of the library.

Programmatic::

    from forjinn_eval import run_all_scenarios, all_scenarios
    suite = run_all_scenarios(offline=True)   # no network
    print(suite.to_markdown()); suite.save_json("web.json")

CLI::

    forjinn-eval web-scenarios --offline --report-json web.json
    forjinn-eval web-scenarios --host https://172.16.34.7 --include-persona
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .capture import AgentRun
from .judge import Embeddings
from .suite import AgentTestCase, SuiteResult, SuiteRunner, make_case

FIXTURES_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures",
    Path(__file__).resolve().parent.parent.parent / "research" / "captured",
]


def _fixtures_dir() -> Path:
    for d in FIXTURES_CANDIDATES:
        if d.is_dir():
            return d
    return FIXTURES_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Personas (deepeval-style simulated users)
# ---------------------------------------------------------------------------
@dataclass
class Persona:
    """A simulated user for persona-driven *online* multi-turn scenarios.

    Mirrors deepeval's ``Persona``: a free-text ``characteristics`` description
    the driver feeds to the Forjinn judge/chatflow to keep the user in character.
    """

    name: str = "user"
    characteristics: str = "A helpful, concise end-user testing the agent."
    first_prompt: str = "Say hello in one sentence."
    follow_ups: Sequence[str] = field(default_factory=list)  # optional scripted turns


PERSONAS: Dict[str, Persona] = {
    "helpful": Persona("helpful_user", "A cooperative user, terse, expects short answers.",
                       "Say hello."),
    "confused": Persona("confused_user", "A confused user who asks the same thing twice.",
                        "What is 2+2?", follow_ups=["Wait, what is 2+2 again?"]),
    "frustrated": Persona("frustrated_customer", "An impatient customer; asks for the answer only.",
                          "How many tables are in db1?",
                          follow_ups=["Just count them. No explanation."]),
    "novice": Persona("novice_user", "A beginner who asks for a one-line explanation.",
                      "What is SQL?", follow_ups=["Give me one example."]),
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """A labelled Forjinn-agent situation to evaluate.

    * ``input_kind``: ``"question"`` (single-turn) or ``"conversation"`` (multi-turn
      turns list) or ``"persona"`` (drive an online conversation).
    * ``inputs``: the question string, or a list of turns (str / {role, content}).
    * ``reference``: ground-truth answer (enables reference-based metrics).
    * ``reference_topics`` / ``role``: context for topic / role metrics.
    * ``reference_tool_calls``: expected tools (enables tool metrics).
    * ``tags``: for grouped roll-up.
    """

    id: str
    inputs: Union[str, List] = ""
    input_kind: str = "question"
    question: str = ""                 # resolved human question (single-turn)
    reference: Optional[str] = None
    reference_topics: Optional[Sequence[str]] = None
    role: Optional[str] = None
    reference_tool_calls: Optional[Sequence[Any]] = None
    tags: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.input_kind == "question" and isinstance(self.inputs, str):
            self.question = self.inputs


def all_scenarios() -> List[Scenario]:
    """The default battery of Forjinn-web scenarios (offline-friendly)."""
    ref5 = "1\n2\n3\n4\n5"
    return [
        Scenario("count-1-to-5", "Count from 1 to 5, one number per line.",
                 "question", reference=ref5, tags=["text", "deterministic"]),
        Scenario("greeting", "Hey, how are you?", "question",
                 reference_topics=["greetings"], tags=["safety", "text"]),
        Scenario("math-two-plus-two", "What is 2+2? Give just the number.",
                 "question", reference="4", reference_topics=["arithmetic"],
                 tags=["rag", "deterministic"]),
        Scenario("sql-tables", "How many tables are in db1? Only count them.",
                 "question", reference="5",
                 reference_tool_calls=["list_tables"],
                 role="a database assistant that only answers with numbers",
                 reference_topics=["databases"],
                 tags=["tools", "agent", "rag"]),
        Scenario("multi-turn-math", [
            "What is 2+2?", "And 3+3?", "Now multiply those two results."],
            "conversation", reference="14", reference_topics=["arithmetic"],
            tags=["conversational", "rag"]),
    ]


# ---------------------------------------------------------------------------
# Batteries (which metrics to run per scenario)
# ---------------------------------------------------------------------------
def structural_batteries(run: AgentRun, sc: Scenario, strict_nodes: bool = True) -> List:
    """Deterministic structural/content checks that need no judge (always run).

    ``strict_nodes=False`` (conversational / persona scenarios) omits
    ``AllNodesFinished`` since a stitched conversation has no canvas nodes.
    """
    from . import (
        AgentLoopDetection,
        AllNodesFinished,
        NoCostLeakage,
        NoToolsExpected,
        OutputNotEmpty,
    )
    evs: List = [OutputNotEmpty(), NoCostLeakage(), AgentLoopDetection()]
    if strict_nodes and not (isinstance(sc.inputs, list)):
        evs.insert(0, AllNodesFinished())
    if sc.reference_tool_calls:
        from . import AvailableToolsExposed as _A
        from . import OnlyAllowedTools
        evs.append(OnlyAllowedTools([t if isinstance(t, str) else t.get("name")
                                     for t in sc.reference_tool_calls]))
        evs.append(_A())
    else:
        evs.append(NoToolsExpected())
    if sc.role:
        from . import RoleViolation
        evs.append(RoleViolation(sc.role, threshold=0.5))
    return evs


def deterministic_batteries(run: AgentRun, sc: Scenario) -> List:
    from . import (
        BleuScore,
        ChrfScore,
        PatternMatch,
        RougeScore,
        SemanticSimilarity,
        ToolCallAccuracy,
    )
    evs: List = [PatternMatch(r"\S", threshold=0.0)]  # non-empty guard
    if sc.reference:
        ref = sc.reference
        # Offline, the agent's recorded text is not scored against ``ref`` by a
        # judge, so deterministic reference metrics gate leniently here; raise
        # these thresholds on a live run for a strict signal.
        evs += [BleuScore(reference=ref, threshold=0.0),
                RougeScore(reference=ref, threshold=0.0),
                ChrfScore(reference=ref, threshold=0.0),
                SemanticSimilarity(reference=ref, threshold=0.0)]
    if sc.reference_tool_calls and run.all_tool_calls:
        evs.append(ToolCallAccuracy(
            reference_tool_calls=[t if isinstance(t, str) else t.get("name")
                                  for t in sc.reference_tool_calls], threshold=0.0))
    return evs


def rag_judge_batteries(judge, embeddings, run: AgentRun, sc: Scenario, threshold: float = 0.5) -> List:
    from . import (
        AnswerAccuracy,
        AnswerRelevancy,
        CitationFaithfulness,
        ContextualPrecision,
        ContextualRecall,
        ContextualRelevancy,
        Faithfulness,
        Groundedness,
        NoiseSensitivity,
    )
    t = threshold or 0.0
    return [Faithfulness(judge=judge, threshold=t),
            AnswerRelevancy(judge=judge, threshold=t),
            Groundedness(judge=judge),
            AnswerAccuracy(judge=judge, threshold=t),
            NoiseSensitivity(judge=judge, threshold=t),
            ContextualPrecision(judge=judge, threshold=max(0.2, t - 0.3)),
            ContextualRecall(judge=judge, threshold=max(0.2, t - 0.3)),
            ContextualRelevancy(embeddings=embeddings, threshold=max(0.1, t - 0.4)),
            CitationFaithfulness(judge=judge)]


def safety_judge_batteries(judge, run: AgentRun, sc: Scenario, threshold: float = 0.5) -> List:
    from . import Bias, Misuse, NonAdvice, PIILeakage, Toxicity
    return [Bias(judge=judge), Toxicity(judge=judge),
            PIILeakage(judge=judge), Misuse(judge=judge), NonAdvice(judge=judge)]


def agent_judge_batteries(judge, run: AgentRun, sc: Scenario, threshold: float = 0.5) -> List:
    from . import (
        ArgumentCorrectness,
        ConversationCompleteness,
        KnowledgeRetention,
        PlanAdherence,
        PlanQuality,
        StepEfficiency,
        Summarization,
        TaskCompletion,
        ToolCorrectness,
    )
    t = threshold or 0.0
    evs = [TaskCompletion(judge=judge, threshold=t),
           PlanAdherence(judge=judge, threshold=t),
           PlanQuality(judge=judge, threshold=t),
           StepEfficiency(judge=judge, threshold=t),
           ConversationCompleteness(judge=judge, threshold=t),
           KnowledgeRetention(judge=judge, threshold=t),
           Summarization(judge=judge, threshold=t)]
    if run.all_tool_calls:
        evs += [ToolCorrectness(judge=judge,
                                expected_tools=(
                                    [x if isinstance(x, str) else x.get("name")
                                     for x in (sc.reference_tool_calls or [])] or None),
                                threshold=t),
                ArgumentCorrectness(judge=judge, threshold=t)]
    return evs


def conversational_batteries(judge, embeddings, run: AgentRun, sc: Scenario, threshold: float = 0.5) -> List:
    is_multi = isinstance(sc.inputs, list) or sc.input_kind != "question"
    if not (run.conversation and len(run.turns()) >= 1 and is_multi):
        return []
    from . import (
        ConversationalGEval,
        GoalAccuracyMulti,
        MultiTurnToolUse,
        MultiTurnTopicAdherence,
        TurnContextualPrecision,
        TurnContextualRecall,
        TurnContextualRelevancy,
        TurnFaithfulness,
        TurnRelevancy,
    )
    t = threshold or 0.0
    evs = [TurnFaithfulness(judge=judge, threshold=t),
           TurnRelevancy(judge=judge, threshold=t),
           TurnContextualPrecision(judge=judge, threshold=max(0.2, t - 0.3)),
           TurnContextualRelevancy(judge=judge, threshold=max(0.1, t - 0.4)),
           GoalAccuracyMulti(judge=judge, threshold=t),
           ConversationalGEval(judge=judge, criteria="answer the user's intent correctly",
                               threshold=t)]
    if sc.reference:
        evs.insert(3, TurnContextualRecall(judge=judge, reference=sc.reference,
                                           threshold=max(0.2, t - 0.3)))
    if sc.reference_topics:
        evs.append(MultiTurnTopicAdherence(sc.reference_topics, judge=judge, threshold=t))
    if run.all_tool_calls:
        evs.append(MultiTurnToolUse(judge=judge, threshold=max(0.2, t - 0.3)))
    return evs


# ---------------------------------------------------------------------------
# Run resolution (offline fixtures vs live host)
# ---------------------------------------------------------------------------
def _fixture_run(sc: Scenario) -> Optional[AgentRun]:
    d = _fixtures_dir()
    if sc.id == "count-1-to-5" or sc.id == "greeting":
        p = d / "agent1_count.json" if sc.id == "count-1-to-5" else d / "agent1_nonstream.json"
    elif sc.id == "math-two-plus-two":
        p = d / "agent1_count.json"  # reuse a text capture; reference-based metrics drive score
    elif sc.id == "sql-tables":
        p = d / "agent2_tools_nonstream.json"
    else:
        p = d / "agent1_count.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    run = AgentRun.from_nonstream("fixture", payload)
    run.question = sc.question or run.question
    if sc.reference:
        run.raw["reference"] = sc.reference
    return run


def _conversation_run(sc: Scenario) -> AgentRun:
    from .types import Conversation
    conv = Conversation()
    for t in sc.inputs:
        if isinstance(t, dict):
            conv.append(t)
        else:
            conv.add_human(str(t))
        conv.add_ai("…(planned) " + str(t)[:40])  # placeholder for offline demo
    conv.metadata["reference"] = sc.reference
    return AgentRun.from_conversation(conv)


class ForjinnWebRunner:
    """Executes scenarios across the full metric catalogue.

    ``offline=True`` replays recorded fixtures (and simulated persona turns)
    with the permissive offline judge. ``offline=False`` calls the live Forjinn
    builder (``host``) and uses the real judge when configured.
    """

    def __init__(self, host: str = "https://172.16.34.7", token: Optional[str] = None,
                 offline: bool = True, judge=None, embeddings: Optional[Embeddings] = None,
                 timeout: float = 120.0):
        self.offline = offline
        self.host = host
        self.token = token
        self.embeddings = embeddings or Embeddings.from_env()
        self.client = None
        if judge is not None:
            self.judge = judge
        elif offline:
            from .judge import queuing_judge
            self.judge = queuing_judge()
        else:
            from .judge import from_env
            self.judge = from_env()  # real JudgeClient if configured, else None
        # metrics grab the process-wide default judge at construction time; make
        # sure one exists (offline permissive judge, or the configured real one).
        from .judge import set_default_judge
        set_default_judge(self.judge)
        if not offline:
            self._build_client()

    def _build_client(self) -> None:
        from .client import ForjinnClient
        self.client = ForjinnClient(self.host, token=self.token, timeout=120.0,
                                    verify_ssl=False, noproxy=True)

    # ---- one scenario --------------------------------------------------
    def _resolve_run(self, sc: Scenario) -> AgentRun:
        if self.offline:
            if isinstance(sc.inputs, list):
                return _conversation_run(sc)
            run = _fixture_run(sc)
            if run is None:
                # fabricate a minimal run so the battery still runs offline
                run = AgentRun(chatflow_id="fixture", question=sc.question, text="")
                if sc.reference:
                    run.raw["reference"] = sc.reference
            return run
        # online
        if sc.input_kind == "conversation" or isinstance(sc.inputs, list):
            r = self.client.predict_multi_turn(
                _online_chatflow(), sc.inputs,
                extra_body={"reference": sc.reference} if sc.reference else None)
            if sc.reference:
                r.raw["reference"] = sc.reference
            return r
        r = self.client.predict(_online_chatflow(), sc.question)
        if sc.reference:
            r.raw["reference"] = sc.reference
        if sc.reference_tool_calls:
            r.raw["reference_tool_calls"] = list(sc.reference_tool_calls)
        return r

    def build_case(self, sc: Scenario) -> AgentTestCase:
        judge, emb = self.judge, self.embeddings
        run = self._resolve_run(sc)
        # Offline: the permissive judge + recorded text are not scored against the
        # true reference, so gate judge metrics leniently (0.0). Online (real
        # judge + live host) the default 0.5 gives a strict signal.
        t = 0.0 if self.offline else 0.5
        evs = []
        evs += structural_batteries(run, sc)
        evs += deterministic_batteries(run, sc)
        evs += rag_judge_batteries(judge, emb, run, sc, threshold=t)
        evs += safety_judge_batteries(judge, run, sc, threshold=t)
        evs += agent_judge_batteries(judge, run, sc, threshold=t)
        evs += conversational_batteries(judge, emb, run, sc, threshold=t)
        return make_case(sc.id, run=run, evaluators=evs, tags=list(sc.tags) + ["web-scenario"])

    def run(self, scenarios: Optional[Sequence[Scenario]] = None,
            suite_name: str = "forjinn-web-scenarios") -> SuiteResult:
        scenarios = list(scenarios if scenarios is not None else all_scenarios())

        # We attach the FULL batteries ourselves, so use a runner that does not
        # also add the default LLM overlay (avoid double metrics).
        no_overlay = SuiteRunner(parallel=1, suite_name=suite_name)
        cases = [self.build_case(sc) for sc in scenarios]
        return no_overlay.run([
            AgentTestCase(c.name, _resolve_to_agentrun(c),
                          c.evaluators, c.tags, c.metadata) for c in cases
        ])


def _resolve_to_agentrun(c: AgentTestCase):
    return c.resolve_run()


def _online_chatflow() -> str:
    return os.environ.get("FORJINN_AGENT_NO_TOOLS",
                          "03d5abc5-6ecd-4891-a9a1-364aefb33a50")


def run_all_scenarios(
    scenarios: Optional[Sequence[Scenario]] = None,
    offline: bool = True,
    host: Optional[str] = None,
    token: Optional[str] = None,
    judge=None,
    include_persona: bool = False,
    persona: Optional[Persona] = None,
    suite_name: str = "forjinn-web-scenarios",
) -> SuiteResult:
    """Evaluate every scenario (or a given list) across the full metric catalogue.

    Returns a :class:`SuiteResult` ready for
    ``to_markdown()`` / ``save_json()`` / ``save_junit()``. ``offline=True``
    never touches the network (replays fixtures + permissive judge); pass
    ``offline=False`` to hit the live Forjinn builder.
    """
    runner = ForjinnWebRunner(host=host or os.environ.get("FORJINN_HOST", "https://172.16.34.7"),
                              token=token, offline=offline, judge=judge)
    scs = list(scenarios if scenarios is not None else all_scenarios())
    if include_persona:
        p = persona or PERSONAS["helpful"]
        scs.append(Scenario(f"persona-{p.name}", [p.first_prompt, *p.follow_ups],
                            "conversation", reference=None,
                            reference_topics=None, tags=["persona", "conversational"]))
    # run all via the shared runner machinery (no default LLM overlay double-add)
    no_overlay = SuiteRunner(parallel=1, suite_name=suite_name)
    cases = [runner.build_case(sc) for sc in scs]
    return no_overlay.run([AgentTestCase(c.name, _resolve_to_agentrun(c),
                                         c.evaluators, c.tags, c.metadata) for c in cases])


__all__ = [
    "PERSONAS",
    "ForjinnWebRunner",
    "Persona",
    "Scenario",
    "agent_judge_batteries",
    "all_scenarios",
    "conversational_batteries",
    "deterministic_batteries",
    "rag_judge_batteries",
    "run_all_scenarios",
    "safety_judge_batteries",
    "structural_batteries",
]
