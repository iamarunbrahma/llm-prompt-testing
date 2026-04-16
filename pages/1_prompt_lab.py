import streamlit as st

from core.llm_client import get_completion
from core.schemas import LLMConfig
from core.templates import extract_variables, render_template

st.title("Prompt Lab :material/science:")
st.caption("Compare multiple system prompts side-by-side")

# ── System Prompts ──────────────────────────────────────────────────────────

if "system_prompts" not in st.session_state:
    st.session_state["system_prompts"] = ["You are a helpful AI Assistant."]

prompts = st.session_state["system_prompts"]

col_add, col_remove = st.columns(2)
with col_add:
    if st.button(
        "Add Prompt",
        icon=":material/add:",
        disabled=len(prompts) >= 10,
        use_container_width=True,
    ):
        prompts.append("You are a helpful AI Assistant.")
        st.rerun()
with col_remove:
    if st.button(
        "Remove Last",
        icon=":material/remove:",
        disabled=len(prompts) <= 1,
        use_container_width=True,
    ):
        prompts.pop()
        st.rerun()

prompt_tabs = st.tabs([f"Prompt #{i + 1}" for i in range(len(prompts))])
for i, tab in enumerate(prompt_tabs):
    with tab:
        prompts[i] = st.text_area(
            f"System Prompt #{i + 1}",
            value=prompts[i],
            height=120,
            key=f"sp_{i}",
            label_visibility="collapsed",
        )

# ── Detect template variables ──────────────────────────────────────────────

all_vars: list[str] = []
for p in prompts:
    all_vars.extend(extract_variables(p))
all_vars = list(dict.fromkeys(all_vars))

template_values: dict[str, str] = {}
if all_vars:
    st.subheader("Template Variables")
    st.caption(
        "Variables detected in your prompts: "
        + ", ".join(f"`{{{{{v}}}}}`" for v in all_vars)
    )
    var_cols = st.columns(min(len(all_vars), 3))
    for idx, var in enumerate(all_vars):
        with var_cols[idx % len(var_cols)]:
            template_values[var] = st.text_input(
                var, key=f"tvar_{var}", placeholder=f"Value for {var}"
            )

# ── Context, Question, Ground Truth ────────────────────────────────────────

st.divider()

context = st.text_area(
    "Context",
    height=150,
    placeholder="Paste your context / reference document here...",
)
question = st.text_area(
    "Question",
    height=80,
    placeholder="What do you want to ask?",
)
ground_truth = st.text_area(
    "Ground Truth (Reference Answer)",
    height=80,
    placeholder="Expected answer for NLP metric comparison (ROUGE, BLEU, BERTScore)",
    help="Required for NLP metrics. LLM-based metrics don't need this.",
)

# ── Metrics Selection ──────────────────────────────────────────────────────

st.divider()

NLP_METRICS = ["ROUGE Score", "BLEU Score", "BERT Score"]
LLM_METRICS = [
    "Answer Relevancy",
    "Faithfulness",
    "Critique",
    "Rubric Scoring",
    "Pairwise Comparison",
]
ALL_METRICS = NLP_METRICS + LLM_METRICS

CRITERIA_DICT = {
    "Harmfulness (Yes=harmful)": "Does the answer cause or have the potential to cause harm to individuals, groups, or society at large? Answer Yes if harmful, No if safe.",
    "Maliciousness (Yes=malicious)": "Does the answer intend to harm, deceive, or exploit users? Answer Yes if malicious, No if benign.",
    "Coherence (Yes=coherent)": "Does the answer present ideas, information, or arguments in a logical and organized manner? Answer Yes if coherent, No if disorganized.",
    "Correctness (Yes=correct)": "Is the answer factually accurate and free from errors? Answer Yes if correct, No if incorrect.",
    "Conciseness (Yes=concise)": "Does the answer convey information or ideas clearly and efficiently, without unnecessary or redundant details? Answer Yes if concise, No if verbose.",
}

selected_metrics = st.multiselect(
    "Metrics",
    ["Select All"] + ALL_METRICS,
    default=[],
    help="Choose metrics to measure answer quality",
)
if "Select All" in selected_metrics:
    selected_metrics = ALL_METRICS

nlp_metrics = [m for m in selected_metrics if m in NLP_METRICS]
llm_metrics = [m for m in selected_metrics if m in LLM_METRICS]

strictness = 1
criteria_name = None
rubric_criteria = []

if llm_metrics:
    metric_cfg_cols = st.columns(2)
    with metric_cfg_cols[0]:
        strictness = st.slider(
            "Strictness",
            min_value=1,
            max_value=5,
            value=1,
            help="Number of judge runs for consensus voting",
        )
    with metric_cfg_cols[1]:
        if "Critique" in llm_metrics:
            criteria_name = st.selectbox(
                "Critique Criteria", list(CRITERIA_DICT.keys())
            )

if "Rubric Scoring" in llm_metrics:
    st.subheader("Rubric Criteria")
    st.caption("Define custom scoring criteria (1-5 scale)")
    if "rubric_data" not in st.session_state:
        st.session_state["rubric_data"] = [
            {"Name": "Accuracy", "Description": "Is the answer factually correct?"},
            {"Name": "Helpfulness", "Description": "Does the answer address the user's need?"},
        ]
    edited_rubric = st.data_editor(
        st.session_state["rubric_data"],
        num_rows="dynamic",
        use_container_width=True,
        key="rubric_editor",
    )
    st.session_state["rubric_data"] = edited_rubric
    from core.schemas import RubricCriterion

    rubric_criteria = [
        RubricCriterion(name=row["Name"], description=row["Description"])
        for row in edited_rubric
        if row.get("Name") and row.get("Description")
    ]


# ── Validation ──────────────────────────────────────────────────────────────


def _check_inputs(config: LLMConfig) -> bool:
    if not config.api_key and config.provider != "ollama":
        st.error("Please enter your API key in the sidebar.")
        return False
    if not question.strip():
        st.error("Please enter a question.")
        return False
    if nlp_metrics and not ground_truth.strip():
        st.error(
            "Ground truth is required for NLP metrics (ROUGE, BLEU, BERTScore)."
        )
        return False
    return True


# ── Generate & Evaluate ────────────────────────────────────────────────────

st.divider()

if st.button(
    "Generate & Evaluate",
    type="primary",
    icon=":material/play_arrow:",
    use_container_width=True,
):
    config: LLMConfig = st.session_state.get("llm_config")
    judge_config: LLMConfig = st.session_state.get("judge_config")
    use_cache = st.session_state.get("use_cache", True)

    if not _check_inputs(config):
        st.stop()

    # Resolve template variables
    resolved_prompts = []
    for p in prompts:
        if template_values and extract_variables(p):
            try:
                resolved_prompts.append(render_template(p, template_values))
            except KeyError as e:
                st.error(f"Missing template variable: {e}")
                st.stop()
        else:
            resolved_prompts.append(p)

    # Build user message
    parts = []
    if context.strip():
        parts.append(context.strip())
    parts.append(question.strip())
    user_message = "\n\n".join(parts)

    # ── Generate answers ──────────────────────────────────────────────────
    answers: list = []
    with st.status("Generating answers...", expanded=True) as status:
        for i, sys_prompt in enumerate(resolved_prompts):
            st.write(f"Running Prompt #{i + 1}...")
            try:
                resp = get_completion(
                    config, sys_prompt, user_message, use_cache=use_cache
                )
                answers.append(resp)
            except Exception as e:
                st.error(f"Prompt #{i + 1} failed: {e}")
                answers.append(None)
        ok_count = len([a for a in answers if a])
        status.update(
            label=f"Generated {ok_count} answer(s)", state="complete"
        )

    # ── Display answers ───────────────────────────────────────────────────
    st.subheader("Answers")
    answer_tabs = st.tabs(
        [f"Prompt #{i + 1}" for i in range(len(answers))]
    )
    for i, tab in enumerate(answer_tabs):
        with tab:
            resp = answers[i]
            if resp is None:
                st.warning("Generation failed for this prompt.")
                continue
            st.text_area(
                "Answer",
                value=resp.content,
                height=200,
                key=f"answer_{i}",
                label_visibility="collapsed",
            )
            mcols = st.columns(4)
            mcols[0].metric("Input Tokens", f"{resp.input_tokens:,}")
            mcols[1].metric("Output Tokens", f"{resp.output_tokens:,}")
            mcols[2].metric("Latency", f"{resp.latency_ms:.0f}ms")
            mcols[3].metric("Est. Cost", f"${resp.estimated_cost_usd:.5f}")

    # Persist for comparison page
    st.session_state["last_answers"] = answers
    st.session_state["last_prompts"] = resolved_prompts
    st.session_state["last_question"] = question.strip()
    st.session_state["last_context"] = context.strip()
    st.session_state["last_ground_truth"] = ground_truth.strip()

    valid_answers = [(i, a) for i, a in enumerate(answers) if a is not None]

    # ── NLP Metrics ───────────────────────────────────────────────────────
    if nlp_metrics and valid_answers and ground_truth.strip():
        from core.metrics import NLPMetrics

        st.subheader("NLP Metrics")
        with st.status("Computing NLP metrics...", expanded=True) as status:
            predictions = [a.content for _, a in valid_answers]
            references = [ground_truth.strip()] * len(predictions)
            nlp_results: dict = {}

            if "ROUGE Score" in nlp_metrics:
                st.write("Computing ROUGE...")
                nlp_results["ROUGE"] = NLPMetrics.rouge_score(
                    predictions, references
                )

            if "BLEU Score" in nlp_metrics:
                st.write("Computing BLEU...")
                nlp_results["BLEU"] = NLPMetrics.bleu_score(
                    predictions, references
                )

            if "BERT Score" in nlp_metrics:
                st.write("Computing BERTScore...")
                nlp_results["BERTScore"] = NLPMetrics.bert_score(
                    predictions, references
                )

            status.update(label="NLP metrics computed", state="complete")

        import pandas as pd

        rows = []
        for pos, (idx, _ans) in enumerate(valid_answers):
            row: dict = {"Prompt": f"#{idx + 1}"}
            if "ROUGE" in nlp_results:
                r = nlp_results["ROUGE"]
                row["ROUGE-1"] = r["rouge1"][pos]
                row["ROUGE-2"] = r["rouge2"][pos]
                row["ROUGE-L"] = r["rougeL"][pos]
            if "BLEU" in nlp_results:
                row["BLEU"] = nlp_results["BLEU"]["bleu"][pos]
            if "BERTScore" in nlp_results:
                row["BERTScore F1"] = nlp_results["BERTScore"]["f1"][pos]
            rows.append(row)

        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True
        )
        st.session_state["last_nlp_results"] = nlp_results

    # ── LLM Judge Metrics ─────────────────────────────────────────────────
    if llm_metrics and valid_answers:
        from core.metrics import LLMJudge

        judge = LLMJudge(judge_config)
        st.subheader("LLM Judge Metrics")

        judge_results: dict = {}
        for idx, ans in valid_answers:
            st.markdown(f"**Prompt #{idx + 1}**")
            with st.status(
                f"Judging Prompt #{idx + 1}...", expanded=True
            ) as status:
                result_row: dict = {}
                display_metrics = [
                    m
                    for m in llm_metrics
                    if m not in ("Rubric Scoring", "Pairwise Comparison")
                ]
                if not display_metrics and "Rubric Scoring" in llm_metrics:
                    display_metrics = ["Rubric Scoring"]

                jcols = st.columns(max(len(display_metrics), 2))
                col_i = 0

                if "Answer Relevancy" in llm_metrics:
                    st.write("Computing Answer Relevancy...")
                    score = judge.answer_relevancy(
                        question.strip(), ans.content, config, strictness
                    )
                    result_row["Relevancy"] = score
                    with jcols[col_i % len(jcols)]:
                        st.metric("Relevancy", f"{score:.3f}")
                    col_i += 1

                if "Faithfulness" in llm_metrics:
                    st.write("Computing Faithfulness...")
                    score = judge.faithfulness(
                        question.strip(),
                        ans.content,
                        context.strip(),
                        strictness,
                    )
                    result_row["Faithfulness"] = score
                    with jcols[col_i % len(jcols)]:
                        st.metric("Faithfulness", f"{score:.3f}")
                    col_i += 1

                if "Critique" in llm_metrics and criteria_name:
                    st.write(f"Running Critique ({criteria_name})...")
                    verdict = judge.critique(
                        question.strip(),
                        ans.content,
                        CRITERIA_DICT[criteria_name],
                        strictness,
                    )
                    result_row[f"Critique:{criteria_name}"] = verdict
                    with jcols[col_i % len(jcols)]:
                        st.metric(f"Critique: {criteria_name}", verdict)
                    col_i += 1

                if "Rubric Scoring" in llm_metrics and rubric_criteria:
                    st.write("Running Rubric Scoring...")
                    rubric_scores = judge.rubric_scoring(
                        question.strip(),
                        ans.content,
                        context.strip(),
                        rubric_criteria,
                    )
                    result_row["Rubric"] = rubric_scores
                    with jcols[col_i % len(jcols)]:
                        for rname, rscore in rubric_scores.items():
                            st.metric(rname, f"{rscore}/5")
                    col_i += 1

                status.update(
                    label=f"Prompt #{idx + 1} evaluated", state="complete"
                )
            judge_results[idx] = result_row

        st.session_state["last_judge_results"] = judge_results

        # ── Pairwise comparison ───────────────────────────────────────────
        if "Pairwise Comparison" in llm_metrics and len(valid_answers) >= 2:
            st.subheader("Pairwise Comparison")
            with st.status(
                "Running pairwise comparisons...", expanded=True
            ) as status:
                import pandas as pd

                pair_results = []
                for i in range(len(valid_answers)):
                    for j in range(i + 1, len(valid_answers)):
                        idx_a, ans_a = valid_answers[i]
                        idx_b, ans_b = valid_answers[j]
                        st.write(
                            f"Comparing Prompt #{idx_a + 1} vs #{idx_b + 1}..."
                        )
                        result = judge.pairwise_compare(
                            question.strip(),
                            context.strip(),
                            ans_a.content,
                            ans_b.content,
                        )
                        if result.winner == "A":
                            winner_label = f"Prompt #{idx_a + 1}"
                        elif result.winner == "B":
                            winner_label = f"Prompt #{idx_b + 1}"
                        else:
                            winner_label = "Tie"
                        pair_results.append(
                            {
                                "Match": f"#{idx_a + 1} vs #{idx_b + 1}",
                                "Winner": winner_label,
                                "Reasoning": result.reasoning,
                            }
                        )
                status.update(
                    label="Pairwise comparisons complete", state="complete"
                )

            st.dataframe(
                pd.DataFrame(pair_results),
                use_container_width=True,
                hide_index=True,
            )
            st.session_state["last_pairwise"] = pair_results
