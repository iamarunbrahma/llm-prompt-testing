import pandas as pd
import streamlit as st

from core.llm_client import get_completion
from core.metrics import LLMJudge, NLPMetrics
from core.schemas import LLMConfig


def _find_col_index(columns: list[str], candidates: list[str]) -> int:
    lower_cols = [c.lower().strip() for c in columns]
    for candidate in candidates:
        if candidate.lower() in lower_cols:
            return lower_cols.index(candidate.lower())
    return 0


st.title("Batch Evaluation :material/table_chart:")
st.caption("Upload a CSV to evaluate prompts across many questions at once")

# ── CSV Upload ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload CSV",
    type="csv",
    help="CSV must contain columns for questions and contexts. A ground_truth column enables NLP metrics.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True, hide_index=True)

# ── Column Mapping ──────────────────────────────────────────────────────────

st.subheader("Column Mapping")
columns = list(df.columns)

map_cols = st.columns(3)
with map_cols[0]:
    question_col = st.selectbox(
        "Question column",
        columns,
        index=_find_col_index(columns, ["question", "questions", "query"]),
    )
with map_cols[1]:
    context_col = st.selectbox(
        "Context column",
        columns,
        index=_find_col_index(columns, ["context", "contexts", "passage"]),
    )
with map_cols[2]:
    gt_options = ["(none)"] + columns
    gt_col = st.selectbox(
        "Ground Truth column (optional)",
        gt_options,
        index=_find_col_index(
            gt_options, ["ground_truth", "groundtruth", "reference", "answer"]
        ),
    )
    has_ground_truth = gt_col != "(none)"

# ── Metrics Selection ──────────────────────────────────────────────────────

st.divider()

NLP_METRICS = ["ROUGE Score", "BLEU Score", "BERT Score"]
LLM_METRICS = ["Answer Relevancy", "Faithfulness", "Critique"]

available_metrics = LLM_METRICS.copy()
if has_ground_truth:
    available_metrics = NLP_METRICS + LLM_METRICS

batch_metrics = st.multiselect(
    "Metrics to compute",
    available_metrics,
    default=["Answer Relevancy", "Faithfulness"] if not has_ground_truth else available_metrics[:3],
)

nlp_batch = [m for m in batch_metrics if m in NLP_METRICS]
llm_batch = [m for m in batch_metrics if m in LLM_METRICS]

CRITERIA_DICT = {
    "Harmfulness": "Does the answer cause or have the potential to cause harm?",
    "Coherence": "Does the answer present ideas in a logical and organized manner?",
    "Correctness": "Is the answer factually accurate and free from errors?",
    "Conciseness": "Does the answer convey information clearly and efficiently?",
}

critique_criteria_name = None
if "Critique" in llm_batch:
    critique_criteria_name = st.selectbox(
        "Critique Criteria", list(CRITERIA_DICT.keys()), key="batch_criteria"
    )

# ── Run ─────────────────────────────────────────────────────────────────────

st.divider()

if st.button(
    "Run Batch Evaluation",
    type="primary",
    icon=":material/play_arrow:",
    use_container_width=True,
):
    config: LLMConfig = st.session_state.get("llm_config")
    judge_config: LLMConfig = st.session_state.get("judge_config")
    use_cache = st.session_state.get("use_cache", True)

    if not config or (not config.api_key and config.provider != "ollama"):
        st.error("Please configure your API key in the sidebar.")
        st.stop()

    prompts = st.session_state.get("system_prompts", ["You are a helpful AI Assistant."])
    num_prompts = len(prompts)

    # Build result columns
    result_cols = ["Question", "Context"]
    if has_ground_truth:
        result_cols.append("Ground Truth")
    result_cols.append("Model")
    for i in range(num_prompts):
        result_cols.append(f"System_Prompt_{i + 1}")
        result_cols.append(f"Answer_{i + 1}")
        result_cols.append(f"Tokens_{i + 1}")
        result_cols.append(f"Cost_{i + 1}")

    if nlp_batch:
        result_cols.extend(nlp_batch)
    for m in llm_batch:
        for i in range(num_prompts):
            if m == "Critique" and critique_criteria_name:
                result_cols.append(f"{m}_{critique_criteria_name}_Prompt{i + 1}")
            else:
                result_cols.append(f"{m}_Prompt{i + 1}")

    results_data: list[dict] = []

    with st.status(
        f"Processing {len(df)} rows...", expanded=True
    ) as status:
        for row_idx, row in df.iterrows():
            st.write(f"Row {row_idx + 1}/{len(df)}")
            q = str(row[question_col])
            ctx = str(row[context_col]) if pd.notna(row[context_col]) else ""
            gt = str(row[gt_col]) if has_ground_truth and pd.notna(row.get(gt_col)) else ""

            parts = []
            if ctx:
                parts.append(ctx)
            parts.append(q)
            user_message = "\n\n".join(parts)

            result_row: dict = {
                "Question": q,
                "Context": ctx,
                "Model": config.model_name,
            }
            if has_ground_truth:
                result_row["Ground Truth"] = gt

            # Generate answers for each prompt
            answer_contents: list[str] = []
            for i, sys_prompt in enumerate(prompts):
                try:
                    resp = get_completion(
                        config, sys_prompt, user_message, use_cache=use_cache
                    )
                    result_row[f"System_Prompt_{i + 1}"] = sys_prompt
                    result_row[f"Answer_{i + 1}"] = resp.content
                    result_row[f"Tokens_{i + 1}"] = f"{resp.input_tokens}+{resp.output_tokens}"
                    result_row[f"Cost_{i + 1}"] = f"${resp.estimated_cost_usd:.5f}"
                    answer_contents.append(resp.content)
                except Exception as e:
                    result_row[f"System_Prompt_{i + 1}"] = sys_prompt
                    result_row[f"Answer_{i + 1}"] = f"ERROR: {e}"
                    result_row[f"Tokens_{i + 1}"] = "0"
                    result_row[f"Cost_{i + 1}"] = "$0"
                    answer_contents.append("")

            # NLP metrics (need ground truth)
            if nlp_batch and gt:
                predictions = answer_contents
                references = [gt] * len(predictions)
                if "ROUGE Score" in nlp_batch:
                    r = NLPMetrics.rouge_score(predictions, references)
                    result_row["ROUGE Score"] = f"R1:{r['rouge1']} R2:{r['rouge2']} RL:{r['rougeL']}"
                if "BLEU Score" in nlp_batch:
                    b = NLPMetrics.bleu_score(predictions, references)
                    result_row["BLEU Score"] = b["bleu"]
                if "BERT Score" in nlp_batch:
                    bs = NLPMetrics.bert_score(predictions, references)
                    result_row["BERT Score"] = bs["mean_f1"]

            # LLM judge metrics
            if llm_batch:
                judge = LLMJudge(judge_config)
                for i, ans_content in enumerate(answer_contents):
                    if not ans_content:
                        continue
                    if "Answer Relevancy" in llm_batch:
                        score = judge.answer_relevancy(q, ans_content, config)
                        result_row[f"Answer Relevancy_Prompt{i + 1}"] = score
                    if "Faithfulness" in llm_batch:
                        score = judge.faithfulness(q, ans_content, ctx)
                        result_row[f"Faithfulness_Prompt{i + 1}"] = score
                    if "Critique" in llm_batch and critique_criteria_name:
                        verdict = judge.critique(
                            q, ans_content, CRITERIA_DICT[critique_criteria_name]
                        )
                        result_row[f"Critique_{critique_criteria_name}_Prompt{i + 1}"] = verdict

            results_data.append(result_row)

        status.update(
            label=f"Processed {len(df)} rows", state="complete"
        )

    # ── Display & Download ────────────────────────────────────────────────
    results_df = pd.DataFrame(results_data)
    st.subheader("Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Report (CSV)",
        csv_data,
        "batch_eval_report.csv",
        "text/csv",
        icon=":material/download:",
        use_container_width=True,
    )

    st.session_state["last_batch_results"] = results_df
