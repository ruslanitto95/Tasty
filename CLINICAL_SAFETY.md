# Clinical safety

**LESS IS BETTER THAN INVENTED.** The AI is never the author of the record: the doctor reviews, edits, approves and copies.

## No-hallucination design
1. The LLM first **extracts facts** (strict JSON schema, `clinical_extraction_v1.md`), it does not write prose.
2. Each fact must carry `evidence_segment_ids` and an `evidence_quote`; `EvidenceValidator` rejects a fact if a segment id is unknown, the quote is not found in those segments, or the value contradicts the evidence. LLM self-confidence is never used as proof.
3. `ConflictResolver`: an explicit patient correction wins («нет, точнее…»); otherwise conflicting facts are dropped and a warning is shown.
4. Formatting uses **validated facts only** (`medical_formatter_v1.md`, NO NEW FACTS). `FinalFactChecker` verifies every sentence (support, numbers, durations, sides, negations, drugs, qualifiers, diagnoses, forbidden phrases) and completeness; on failure one repair is attempted, then the deterministic formatter is used with a warning.
5. If the LLM output is invalid after repairs: «Не удалось сформировать структурированный документ.» — no fallback text is invented; the transcript stays available.

## Critical rules (deterministic)
| Rule | Implementation |
|---|---|
| Absence of information never becomes a negation | `negative` facts require explicit absence wording in the evidence; «не знаю/не помню/не мерил» is uncertainty, not denial |
| Negation preserved | positive facts whose evidence negates them are rejected; statements must keep the negation |
| Numbers | temperatures (incl. «тридцать семь и пять», «37 и 5»), durations (неделю = 7 дней, never weeks↔days), doses must match the evidence |
| Laterality | claimed side ⊆ side stated for that symptom; swaps and narrowing are rejected |
| Medications | names are not guessed; a drug absent from speech is rejected; an LLM "correction" of a misrecognised name is reverted to the spoken form with a «возможно, Назонекс?» warning; «какие-то капли» → «капли, название не уточнено» |
| Qualifiers | «гнойные», «сильная», «двусторонняя»… must be in the evidence |
| Diagnoses / advice | never in complaints/history; doctor hypotheses («похоже на синусит») and patient questions («мне антибиотик надо?») are not facts |
| Unrelated past history | «в детстве была ангина» stays out of the current history |
| Relative dates | «с понедельника» is kept as said; not converted to a duration by guessing |

Warnings («ПРОВЕРИТЬ») are shown to the doctor and never inserted into the document. Every accepted fact keeps `fact → segment → time range → quote`, available via «Показать источник».

## Regression suites
`tests/clinical` holds ≥200 synthetic dialogues (ENT focus + therapy + paediatrics) including dedicated negation, number, laterality, medication and hallucination suites; false-positive facts must be zero. See [TESTING.md](TESTING.md).

## Known limitations
- Speakers are not diarised; the LLM infers question/answer from context.
- Quality of extraction depends on the configured LLM; the deterministic layer bounds the damage (it can drop good facts, it is designed not to let invented ones through) but cannot add missed facts.
- Rules are tuned for Russian outpatient ENT conversations.
