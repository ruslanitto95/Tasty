"""Transcript -> facts -> validation -> conflict resolution -> formatting -> fact check."""

from __future__ import annotations

import logging
import threading
from datetime import date

from mva.clinical.conflict_resolution import ConflictResolver
from mva.clinical.extraction import PROMPT as EXTRACTION_PROMPT
from mva.clinical.extraction import FactExtractor
from mva.clinical.fact_check import FinalFactChecker
from mva.clinical.formatter import DeterministicFormatter, LLMFormatter
from mva.clinical.lexicon import Lexicon, load_lexicon
from mva.clinical.schemas import DocumentDraft, ExtractionResult, FormattedDocument, ReviewWarning
from mva.clinical.text import split_sentences
from mva.clinical.validation import EvidenceValidator
from mva.llm.base import LLMError, LLMProvider
from mva.transcription.models import Transcript

log = logging.getLogger(__name__)


class ClinicalPipeline:
    def __init__(
        self,
        llm: LLMProvider,
        lexicon: Lexicon | None = None,
        use_llm_formatter: bool = True,
    ) -> None:
        self.lexicon = lexicon or load_lexicon()
        self.extractor = FactExtractor(llm)
        self.validator = EvidenceValidator(self.lexicon)
        self.resolver = ConflictResolver(self.lexicon)
        self.deterministic = DeterministicFormatter()
        self.llm_formatter = LLMFormatter(llm) if use_llm_formatter else None
        self.checker = FinalFactChecker(self.lexicon)

    def run(
        self,
        transcript: Transcript,
        visit_date: date | None = None,
        cancel: threading.Event | None = None,
        extraction: ExtractionResult | None = None,
    ) -> DocumentDraft:
        if transcript.is_empty():
            return DocumentDraft(
                warnings=[
                    ReviewWarning(
                        code="no_speech", message="Речь не распознана — документ не сформирован."
                    )
                ]
            )
        extraction = extraction or self.extractor.extract(transcript, visit_date, cancel)
        return self.build(transcript, extraction, cancel)

    def build(
        self,
        transcript: Transcript,
        extraction: ExtractionResult,
        cancel: threading.Event | None = None,
    ) -> DocumentDraft:
        outcome = self.validator.validate(extraction.facts, transcript)
        facts, conflict_warnings = self.resolver.resolve(outcome.accepted, transcript)
        warnings = _unique(outcome.warnings + conflict_warnings)
        order = {s.id: s.seq for s in transcript.segments}
        det_doc, _ = self.deterministic.format(facts, order)
        det_violations = self.checker.check(det_doc, facts)
        if det_violations:
            # Should not happen; strip offending sentences rather than show them.
            log.error(
                "Deterministic formatter failed fact check (%d violations)", len(det_violations)
            )
            det_doc = _strip(det_doc, {v.sentence for v in det_violations})
        doc, formatter = det_doc, "deterministic"
        if self.llm_formatter is not None and facts:
            doc, formatter, extra = self._llm_format(facts, det_doc, cancel)
            warnings.extend(extra)
        if not doc.complaints_text and not doc.history_text:
            warnings.append(
                ReviewWarning(
                    code="insufficient_data",
                    message="Недостаточно данных для формирования жалоб и анамнеза.",
                )
            )
        return DocumentDraft(
            complaints_text=doc.complaints_text,
            history_text=doc.history_text,
            warnings=warnings,
            accepted_facts=facts,
            rejected_facts=outcome.rejected,
            trace=self.checker.trace(doc, facts),
            formatter=formatter,
            extraction_prompt_version=EXTRACTION_PROMPT,
            formatter_prompt_version=LLMFormatter.PROMPT
            if formatter == "llm"
            else "deterministic_v1",
        )

    def _llm_format(
        self, facts: list, fallback: FormattedDocument, cancel: threading.Event | None
    ) -> tuple[FormattedDocument, str, list[ReviewWarning]]:
        assert self.llm_formatter is not None
        try:
            doc = self.llm_formatter.format(facts, cancel)
            violations = self.checker.check(doc, facts)
            if violations:
                log.warning("Formatter output failed fact check (%d); repairing", len(violations))
                doc = self.llm_formatter.format(
                    facts, cancel, [v.describe() for v in violations], doc
                )
                violations = self.checker.check(doc, facts)
            if not violations:
                return doc, "llm", []
            log.warning(
                "Formatter repair failed (%d violations); using deterministic text", len(violations)
            )
        except LLMError as exc:
            log.warning("LLM formatter unavailable (%s); using deterministic text", exc.code)
        return (
            fallback,
            "deterministic",
            [
                ReviewWarning(
                    code="formatter_fallback",
                    message="Использован упрощённый текст: стилистическая обработка не прошла проверку фактов.",
                )
            ],
        )


def _strip(doc: FormattedDocument, bad: set[str]) -> FormattedDocument:
    def keep(text: str) -> str:
        return " ".join(s for s in split_sentences(text) if s not in bad)

    return FormattedDocument(
        complaints_text=keep(doc.complaints_text), history_text=keep(doc.history_text)
    )


def _unique(warnings: list[ReviewWarning]) -> list[ReviewWarning]:
    seen: set[tuple[str, str]] = set()
    out = []
    for w in warnings:
        key = (w.code, w.message)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out
