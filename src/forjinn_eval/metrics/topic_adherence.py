"""ragas TopicAdherence: P/R/F1 over (topic answered AND on-topic)."""
from __future__ import annotations

from typing import Optional, Sequence

from ..capture import AgentRun
from ..judge import JudgeClient, as_bool, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip, _trace_text


class TopicAdherence(Evaluator):
    name = "topic_adherence"
    kind = "rag"

    EXTRACT_TOPICS = (
        "Given an interaction between Human and AI, extract the topics from the Human's input. Return "
        'JSON {"topics": [str, ...]}.\nConversation:\n{conversation}\nJSON:'
    )

    REFUSED_TO_ANSWER = (
        "Given a topic, classify whether the AI REFUSED to answer the question about that topic. Return "
        'JSON {"refused_to_answer": true|false}.\nConversation:\n{conversation}\nTopic: {topic}\nJSON:'
    )

    CLASSIFY_TOPICS = (
        "Given a set of reference topics and a list of extracted topics, for each extracted topic decide "
        'whether it falls into ANY of the reference topics. Return JSON {"classifications": [true|false, '
        "...]} with one boolean per extracted topic.\nReference topics: {reference_topics}\n"
        "Topics: {topics}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 reference_topics: Optional[Sequence[str]] = None, mode: str = "f1",
                 name: Optional[str] = None, kind: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.reference_topics = list(reference_topics or [])
        self.mode = mode
        if name:
            self.name = name
        if kind:
            self.kind = kind

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not self.reference_topics:
            return _nan_or_skip(self.name, "no reference_topics supplied")
        conv = _trace_text(run)
        ext = self.judge.complete_json(_judge_user(_f(self.EXTRACT_TOPICS, conversation=conv)))
        topics = (ext or {}).get("topics", [])
        if not topics:
            return _cr(self.name, 0.0, self.threshold, details={"topics": []})
        answered = []
        for t in topics:
            o = self.judge.complete_json(_judge_user(_f(self.REFUSED_TO_ANSWER, conversation=conv, topic=t)))
            answered.append(not as_bool((o or {}).get("refused_to_answer")))
        cls = self.judge.complete_json(_judge_user(_f(
            self.CLASSIFY_TOPICS, reference_topics=_fmt_list(self.reference_topics),
            topics=_fmt_list(topics))))
        cls_list = (cls or {}).get("classifications", [])
        classifications = [as_bool(x) for x in cls_list]
        while len(classifications) < len(topics):
            classifications.append(False)
        classifications = classifications[: len(topics)]
        answered_b = [as_bool(a) for a in answered]
        tp = sum(1 for a, c in zip(answered_b, classifications) if a and c)
        fp = sum(1 for a, c in zip(answered_b, classifications) if a and not c)
        fn = sum(1 for a, c in zip(answered_b, classifications) if (not a) and c)
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        if self.mode == "precision":
            score = p
        elif self.mode == "recall":
            score = r
        else:
            score = 2 * p * r / (p + r + 1e-10)
        return _cr(self.name, score, self.threshold, details={"tp": tp, "fp": fp, "fn": fn, "topics": topics})


__all__ = ["TopicAdherence"]
