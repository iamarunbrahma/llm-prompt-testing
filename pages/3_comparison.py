import pandas as pd
import streamlit as st

st.title("Comparison :material/compare:")
st.caption("Visualize and compare results from Prompt Lab or Batch Evaluation")

# ── Check for available data ────────────────────────────────────────────────

answers = st.session_state.get("last_answers")
prompts = st.session_state.get("last_prompts")
nlp_results = st.session_state.get("last_nlp_results")
judge_results = st.session_state.get("last_judge_results")
pairwise_results = st.session_state.get("last_pairwise")
batch_results = st.session_state.get("last_batch_results")

has_prompt_lab_data = answers and prompts
has_batch_data = batch_results is not None

if not has_prompt_lab_data and not has_batch_data:
    st.info(
        "No results to display yet. Run an evaluation in **Prompt Lab** "
        "or **Batch Eval** first, then come back here."
    )
    st.stop()

# ── Data source selector ────────────────────────────────────────────────────

sources = []
if has_prompt_lab_data:
    sources.append("Prompt Lab")
if has_batch_data:
    sources.append("Batch Eval")

source = st.pills("Data source", sources, default=sources[0])

# ═══════════════════════════════════════════════════════════════════════════
# Prompt Lab Results
# ═══════════════════════════════════════════════════════════════════════════

if source == "Prompt Lab" and has_prompt_lab_data:
    valid_answers = [(i, a) for i, a in enumerate(answers) if a is not None]

    if not valid_answers:
        st.warning("All answers failed to generate.")
        st.stop()

    # ── Cost Summary ──────────────────────────────────────────────────────
    st.subheader("Cost & Performance Summary")
    summary_cols = st.columns(4)

    total_input = sum(a.input_tokens for _, a in valid_answers)
    total_output = sum(a.output_tokens for _, a in valid_answers)
    total_cost = sum(a.estimated_cost_usd for _, a in valid_answers)
    avg_latency = (
        sum(a.latency_ms for _, a in valid_answers) / len(valid_answers)
    )

    summary_cols[0].metric("Total Input Tokens", f"{total_input:,}")
    summary_cols[1].metric("Total Output Tokens", f"{total_output:,}")
    summary_cols[2].metric("Total Cost", f"${total_cost:.5f}")
    summary_cols[3].metric("Avg Latency", f"{avg_latency:.0f}ms")

    # Per-prompt breakdown
    st.subheader("Per-Prompt Breakdown")
    breakdown_data = []
    for idx, ans in valid_answers:
        breakdown_data.append(
            {
                "Prompt": f"#{idx + 1}",
                "Input Tokens": ans.input_tokens,
                "Output Tokens": ans.output_tokens,
                "Latency (ms)": round(ans.latency_ms),
                "Cost ($)": round(ans.estimated_cost_usd, 5),
            }
        )
    st.dataframe(
        pd.DataFrame(breakdown_data),
        use_container_width=True,
        hide_index=True,
    )

    # ── NLP Metrics Chart ─────────────────────────────────────────────────
    if nlp_results:
        st.subheader("NLP Metrics Comparison")

        chart_data = {}
        prompt_labels = [f"Prompt #{idx + 1}" for idx, _ in valid_answers]

        if "ROUGE" in nlp_results:
            r = nlp_results["ROUGE"]
            chart_data["ROUGE-1"] = [r["rouge1"]] * len(valid_answers)
            chart_data["ROUGE-2"] = [r["rouge2"]] * len(valid_answers)
            chart_data["ROUGE-L"] = [r["rougeL"]] * len(valid_answers)

        if "BLEU" in nlp_results:
            chart_data["BLEU"] = [nlp_results["BLEU"]["bleu"]] * len(
                valid_answers
            )

        if "BERTScore" in nlp_results:
            chart_data["BERTScore F1"] = nlp_results["BERTScore"]["f1"]

        if chart_data:
            chart_df = pd.DataFrame(chart_data, index=prompt_labels)
            st.bar_chart(chart_df)

    # ── LLM Judge Metrics Chart ───────────────────────────────────────────
    if judge_results:
        st.subheader("LLM Judge Metrics Comparison")

        judge_rows = []
        for idx, metrics in judge_results.items():
            row = {"Prompt": f"#{idx + 1}"}
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    row[key] = val
                elif isinstance(val, dict):
                    for k, v in val.items():
                        row[k] = v
                else:
                    row[key] = val
            judge_rows.append(row)

        judge_df = pd.DataFrame(judge_rows)
        st.dataframe(
            judge_df, use_container_width=True, hide_index=True
        )

        # Bar chart for numeric columns only
        numeric_cols = judge_df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            chart_df = judge_df.set_index("Prompt")[numeric_cols]
            st.bar_chart(chart_df)

    # ── Pairwise Results ──────────────────────────────────────────────────
    if pairwise_results:
        st.subheader("Pairwise Comparison Results")
        st.dataframe(
            pd.DataFrame(pairwise_results),
            use_container_width=True,
            hide_index=True,
        )

    # ── Export All Results ────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")

    export_data: dict = {
        "question": st.session_state.get("last_question", ""),
        "context": st.session_state.get("last_context", ""),
        "ground_truth": st.session_state.get("last_ground_truth", ""),
        "prompts": prompts,
        "answers": [
            {
                "prompt_index": i,
                "content": a.content,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "latency_ms": a.latency_ms,
                "cost_usd": a.estimated_cost_usd,
            }
            for i, a in valid_answers
        ],
    }
    if nlp_results:
        export_data["nlp_metrics"] = nlp_results
    if judge_results:
        export_data["judge_metrics"] = {
            str(k): v for k, v in judge_results.items()
        }
    if pairwise_results:
        export_data["pairwise"] = pairwise_results

    import json

    json_str = json.dumps(export_data, indent=2, default=str)
    st.download_button(
        "Download Full Results (JSON)",
        json_str,
        "prompt_lab_results.json",
        "application/json",
        icon=":material/download:",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Batch Eval Results
# ═══════════════════════════════════════════════════════════════════════════

if source == "Batch Eval" and has_batch_data:
    st.subheader("Batch Evaluation Results")
    st.dataframe(batch_results, use_container_width=True, hide_index=True)

    # Numeric columns for charting
    numeric_cols = batch_results.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        st.subheader("Metric Distribution")
        selected_col = st.selectbox("Metric to visualize", list(numeric_cols))
        if selected_col:
            st.bar_chart(batch_results[selected_col])

    st.divider()
    csv_data = batch_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Batch Results (CSV)",
        csv_data,
        "batch_results.csv",
        "text/csv",
        icon=":material/download:",
        use_container_width=True,
    )
