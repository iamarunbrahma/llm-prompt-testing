from __future__ import annotations

import re
from collections import Counter

import evaluate
import numpy as np

from core.llm_client import cosine_similarity, get_completion, get_embedding
from core.schemas import ComparisonResult, LLMConfig, RubricCriterion


# ═══════════════════════════════════════════════════════════════════════════
# NLP Metrics — compare generated answers against ground truth references
# ═══════════════════════════════════════════════════════════════════════════


class NLPMetrics:

    @staticmethod
    def rouge_score(
        predictions: list[str], references: list[str]
    ) -> dict:
        rouge = evaluate.load("rouge")
        # Compute per-answer ROUGE scores for meaningful prompt comparison
        per_answer = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = rouge.compute(predictions=[pred], references=[ref])
            per_answer["rouge1"].append(round(result["rouge1"], 3))
            per_answer["rouge2"].append(round(result["rouge2"], 3))
            per_answer["rougeL"].append(round(result["rougeL"], 3))
        return {
            "rouge1": per_answer["rouge1"],
            "rouge2": per_answer["rouge2"],
            "rougeL": per_answer["rougeL"],
            "mean_rouge1": round(np.mean(per_answer["rouge1"]), 3),
            "mean_rouge2": round(np.mean(per_answer["rouge2"]), 3),
            "mean_rougeL": round(np.mean(per_answer["rougeL"]), 3),
        }

    @staticmethod
    def bleu_score(
        predictions: list[str], references: list[str]
    ) -> dict:
        bleu = evaluate.load("bleu")
        # Compute per-answer BLEU scores (sentence-level)
        per_answer = []
        for pred, ref in zip(predictions, references):
            try:
                result = bleu.compute(predictions=[pred], references=[[ref]])
                per_answer.append(round(result["bleu"], 3))
            except ZeroDivisionError:
                # BLEU can fail on very short texts
                per_answer.append(0.0)
        return {
            "bleu": per_answer,
            "mean_bleu": round(np.mean(per_answer), 3),
        }

    @staticmethod
    def bert_score(
        predictions: list[str],
        references: list[str],
        model_type: str = "distilbert-base-uncased",
    ) -> dict:
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type=model_type,
        )
        f1_scores = [round(s, 3) for s in results["f1"]]
        return {"f1": f1_scores, "mean_f1": round(np.mean(f1_scores), 3)}


# ═══════════════════════════════════════════════════════════════════════════
# LLM Judge — uses a separate judge model for evaluation (never mutates config)
# ═══════════════════════════════════════════════════════════════════════════


class LLMJudge:

    def __init__(self, judge_config: LLMConfig):
        self.config = judge_config

    def _judge_call(self, system_prompt: str, user_message: str) -> str:
        resp = get_completion(
            self.config, system_prompt, user_message, use_cache=False
        )
        return resp.content

    # ── Answer Relevancy ──────────────────────────────────────────────────

    def answer_relevancy(
        self,
        question: str,
        answer: str,
        generation_config: LLMConfig,
        strictness: int = 1,
    ) -> float:
        relevancy_prompt = """Generate a question for the given answer. Only output the question, nothing else.

Examples:
Answer: The first ODI Cricket World Cup was held in 1975, and the West Indies cricket team won the tournament.
Question: Which team won the first ODI Cricket World Cup and in which year?

Answer: The first president of the United States was George Washington, who became president in 1789.
Question: Who was the first president of the United States and when did he become president?

Generate a question that is relevant to the following answer."""

        # Cache the original question embedding (constant across strictness runs)
        try:
            q_vec = get_embedding(question, generation_config)
        except Exception:
            # If embedding fails (e.g., provider doesn't support it),
            # fall back to the judge config (may use a different provider)
            q_vec = get_embedding(question, self.config)

        scores = []
        for _ in range(strictness):
            generated_question = self._judge_call(relevancy_prompt, answer)
            try:
                gq_vec = get_embedding(
                    generated_question, generation_config
                )
            except Exception:
                gq_vec = get_embedding(generated_question, self.config)
            scores.append(cosine_similarity(q_vec, gq_vec))

        return round(float(np.mean(scores)), 3)

    # ── Faithfulness ──────────────────────────────────────────────────────

    def faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
        strictness: int = 1,
    ) -> float:
        if not context.strip():
            return 0.0

        # Step 1: Extract statements from the answer
        stmt_prompt = """Given a question and answer, extract factual statements from the answer.
Output each statement on a new line, numbered.

Example:
Question: Who is Sachin Tendulkar?
Answer: Sachin Tendulkar is a former Indian cricketer widely regarded as one of the greatest batsmen in cricket history. He is often referred to as the "Little Master."
Statements:
1. Sachin Tendulkar is a former Indian cricketer.
2. Sachin Tendulkar is widely regarded as one of the greatest batsmen in cricket history.
3. He is often referred to as the "Little Master."

Extract statements from the following:"""

        stmt_input = f"Question: {question}\nAnswer: {answer}\nStatements:"

        # Step 2: NLI — check each statement against context
        nli_system = "You are a careful fact-checker. For each numbered statement, determine if it is supported by the given context. Reply with ONLY the statement number and verdict."
        nli_template = """Context:
{context}

Statements:
{statements}

For each statement, respond with EXACTLY this format (one per line):
1. Yes
2. No
3. Yes
...and so on. Output NOTHING else — no explanations, no reasoning, just the number and Yes/No."""

        # Regex to match verdict lines like "1. Yes", "2. No", "3: Yes", etc.
        verdict_pattern = re.compile(
            r"^\s*\d+[\.\):\s]+\s*(yes|no)\s*\.?\s*$", re.IGNORECASE
        )

        all_scores: list[float] = []
        for _ in range(strictness):
            statements_raw = self._judge_call(stmt_prompt, stmt_input)
            # Parse numbered statements
            statements = []
            for line in statements_raw.strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned and len(cleaned) > 3:
                    statements.append(cleaned)

            if not statements:
                all_scores.append(0.0)
                continue

            numbered = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(statements)
            )
            nli_input = nli_template.format(
                context=context, statements=numbered
            )
            nli_result = self._judge_call(nli_system, nli_input)

            # Parse verdict lines strictly
            yes_count = 0
            no_count = 0
            for line in nli_result.strip().split("\n"):
                match = verdict_pattern.match(line)
                if match:
                    if match.group(1).lower() == "yes":
                        yes_count += 1
                    else:
                        no_count += 1

            total = yes_count + no_count

            # Fallback: if strict parsing found nothing, try looser matching
            # but only on lines that are very short (likely just verdicts)
            if total == 0:
                for line in nli_result.strip().split("\n"):
                    stripped = line.strip().lower().rstrip(".")
                    if stripped in ("yes", "no"):
                        if stripped == "yes":
                            yes_count += 1
                        else:
                            no_count += 1
                total = yes_count + no_count

            if total == 0:
                all_scores.append(0.0)
            else:
                all_scores.append(yes_count / total)

        return round(float(np.mean(all_scores)), 3)

    # ── Critique ──────────────────────────────────────────────────────────

    def critique(
        self,
        question: str,
        answer: str,
        criteria: str,
        strictness: int = 1,
    ) -> str:
        critique_prompt = """Given a question and answer, evaluate the answer using ONLY the given criteria.
Think step by step providing reasoning, then conclude with a final verdict.

Your final line MUST be exactly one of:
Verdict: Yes
Verdict: No

Example:
Question: Who was the US president during World War 2?
Answer: Franklin D. Roosevelt served as President from 1933 until his death in 1945.
Criteria: Is the output written in perfect grammar?
Reasoning: The answer uses proper sentence structure and correct grammar throughout.
Verdict: Yes"""

        critique_input = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Criteria: {criteria}\n"
            f"Reasoning:"
        )

        responses: list[int] = []
        for _ in range(strictness):
            result = self._judge_call(critique_prompt, critique_input)
            # Parse the final verdict line strictly
            verdict = 0
            for line in reversed(result.strip().split("\n")):
                line_lower = line.strip().lower()
                if line_lower.startswith("verdict:"):
                    verdict_text = line_lower.replace("verdict:", "").strip()
                    if verdict_text.startswith("yes"):
                        verdict = 1
                    break
                # Also accept bare Yes/No as last line
                if line_lower.rstrip(".") in ("yes", "no"):
                    if line_lower.rstrip(".") == "yes":
                        verdict = 1
                    break
            responses.append(verdict)

        majority = Counter(responses).most_common(1)[0][0]
        return "Yes" if majority == 1 else "No"

    # ── Rubric Scoring ────────────────────────────────────────────────────

    def rubric_scoring(
        self,
        question: str,
        answer: str,
        context: str,
        rubric: list[RubricCriterion],
    ) -> dict[str, int]:
        criteria_text = "\n".join(
            f"- {c.name} ({c.scale_min}-{c.scale_max}): {c.description}"
            for c in rubric
        )

        scoring_prompt = f"""Score the answer on each criterion below using an integer score.

Criteria:
{criteria_text}

Example output format (one criterion per line, nothing else):
Accuracy: 4
Helpfulness: 3
Clarity: 5

Now score the following answer. Output ONLY criterion names and integer scores, one per line. No explanations."""

        scoring_input = (
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n\n"
            f"Scores:"
        )

        result = self._judge_call(scoring_prompt, scoring_input)

        scores: dict[str, int] = {}
        for criterion in rubric:
            pattern = re.compile(
                rf"{re.escape(criterion.name)}\s*:\s*(\d+)", re.IGNORECASE
            )
            match = pattern.search(result)
            if match:
                val = int(match.group(1))
                val = max(criterion.scale_min, min(val, criterion.scale_max))
                scores[criterion.name] = val
            else:
                # Fallback: try matching just a number near the criterion name
                fallback = re.compile(
                    rf"{re.escape(criterion.name)}[^\d]*(\d+)", re.IGNORECASE
                )
                fb_match = fallback.search(result)
                if fb_match:
                    val = int(fb_match.group(1))
                    val = max(
                        criterion.scale_min, min(val, criterion.scale_max)
                    )
                    scores[criterion.name] = val
                else:
                    scores[criterion.name] = criterion.scale_min

        return scores

    # ── Pairwise Comparison ───────────────────────────────────────────────

    def _parse_winner(self, result: str) -> tuple[str, str]:
        """Parse winner and reasoning from judge output."""
        result_lower = result.strip().lower()
        if "winner: a" in result_lower:
            winner = "A"
        elif "winner: b" in result_lower:
            winner = "B"
        else:
            winner = "tie"
        lines = result.strip().split("\n")
        reasoning_lines = [
            line
            for line in lines
            if not line.strip().lower().startswith("winner:")
        ]
        reasoning = " ".join(reasoning_lines).strip()
        return winner, reasoning

    def pairwise_compare(
        self,
        question: str,
        context: str,
        answer_a: str,
        answer_b: str,
        criteria: str = "overall quality, accuracy, and helpfulness",
    ) -> ComparisonResult:
        compare_template = """Compare two answers to the same question. Judge based on: {criteria}.

Question: {question}
Context: {context}

Answer A:
{first}

Answer B:
{second}

First explain your reasoning (2-3 sentences), then on the final line write EXACTLY one of: "Winner: A", "Winner: B", or "Winner: Tie"."""

        system = "You are a fair and impartial judge. Evaluate solely on merit, not position."

        # Run 1: A first, B second (original order)
        prompt_1 = compare_template.format(
            criteria=criteria,
            question=question,
            context=context,
            first=answer_a,
            second=answer_b,
        )
        result_1 = self._judge_call(system, prompt_1)
        winner_1, reasoning_1 = self._parse_winner(result_1)

        # Run 2: B first, A second (swapped to debias position preference)
        prompt_2 = compare_template.format(
            criteria=criteria,
            question=question,
            context=context,
            first=answer_b,
            second=answer_a,
        )
        result_2 = self._judge_call(system, prompt_2)
        winner_2_raw, reasoning_2 = self._parse_winner(result_2)
        # Flip the swapped result back to original labels
        if winner_2_raw == "A":
            winner_2 = "B"  # A in swapped = original B
        elif winner_2_raw == "B":
            winner_2 = "A"  # B in swapped = original A
        else:
            winner_2 = "tie"

        # Consensus: both runs must agree, otherwise it's a tie
        if winner_1 == winner_2:
            final_winner = winner_1
            reasoning = reasoning_1
        else:
            final_winner = "tie"
            reasoning = (
                f"Position-debiased result: Run 1 picked {winner_1}, "
                f"Run 2 (swapped) picked {winner_2}. No consensus — tie. "
                f"Run 1 reasoning: {reasoning_1}"
            )

        return ComparisonResult(winner=final_winner, reasoning=reasoning)
