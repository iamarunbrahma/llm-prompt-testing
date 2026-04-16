import streamlit as st

from core.schemas import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDER_MODELS, LLMConfig

st.set_page_config(
    page_title="Prompt Testing v2",
    page_icon=":material/science:",
    layout="wide",
)

# ── Navigation ──────────────────────────────────────────────────────────────

prompt_lab = st.Page(
    "pages/1_prompt_lab.py", title="Prompt Lab", icon=":material/science:"
)
batch_eval = st.Page(
    "pages/2_batch_eval.py", title="Batch Eval", icon=":material/table_chart:"
)
comparison = st.Page(
    "pages/3_comparison.py", title="Comparison", icon=":material/compare:"
)

pg = st.navigation(
    {"Testing": [prompt_lab, batch_eval], "Analysis": [comparison]}
)

# ── Sidebar: Provider & Model ───────────────────────────────────────────────

st.sidebar.header("Configuration", divider="rainbow")

providers = list(PROVIDER_MODELS.keys()) + ["other"]
provider = st.sidebar.pills(
    "Provider",
    providers,
    default=DEFAULT_PROVIDER,
    format_func=str.capitalize,
)
if provider is None:
    provider = DEFAULT_PROVIDER

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Enter your API key",
    help="Required for cloud providers. Not needed for local Ollama.",
)

models_for_provider = PROVIDER_MODELS.get(provider, [])
use_custom_model = st.sidebar.toggle("Custom model name", value=not models_for_provider)

if use_custom_model or not models_for_provider:
    model_name = st.sidebar.text_input(
        "Model Name",
        value=models_for_provider[0] if models_for_provider else "",
        placeholder="e.g. gpt-4o, claude-sonnet-4-20250514",
    )
else:
    model_name = st.sidebar.selectbox("Model", models_for_provider)

# ── Sidebar: Hyperparameters ────────────────────────────────────────────────

st.sidebar.divider()

temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=2.0, step=0.01, value=0.0
)
top_p = st.sidebar.slider(
    "Top P", min_value=0.0, max_value=1.0, step=0.01, value=1.0
)
max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=10, max_value=4096, value=512
)

show_penalties = st.sidebar.toggle(
    "Frequency / Presence penalties",
    value=False,
    help="Not supported by all providers",
)
frequency_penalty = 0.0
presence_penalty = 0.0
if show_penalties:
    frequency_penalty = st.sidebar.slider(
        "Frequency Penalty", min_value=0.0, max_value=2.0, step=0.01, value=0.0
    )
    presence_penalty = st.sidebar.slider(
        "Presence Penalty", min_value=0.0, max_value=2.0, step=0.01, value=0.0
    )

# ── Build config ────────────────────────────────────────────────────────────

config = LLMConfig(
    provider=provider,
    model_name=model_name or DEFAULT_MODEL,
    api_key=api_key,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
)
st.session_state["llm_config"] = config

# ── Sidebar: Judge Model Config ─────────────────────────────────────────────

with st.sidebar.expander("Judge Model Settings", icon=":material/gavel:"):
    st.caption("Model used for LLM-based evaluation metrics")
    judge_provider = st.pills(
        "Judge Provider",
        providers,
        default=provider,
        format_func=str.capitalize,
        key="judge_provider_pills",
    )
    if judge_provider is None:
        judge_provider = provider

    judge_models = PROVIDER_MODELS.get(judge_provider, [])
    if judge_models:
        judge_model = st.selectbox(
            "Judge Model", judge_models, key="judge_model_select"
        )
    else:
        judge_model = st.text_input(
            "Judge Model Name",
            placeholder="e.g. gpt-4o-mini",
            key="judge_model_input",
        )

    judge_api_key = st.text_input(
        "Judge API Key",
        type="password",
        placeholder="Same as above if blank",
        key="judge_api_key_input",
    )

judge_config = LLMConfig(
    provider=judge_provider,
    model_name=judge_model or DEFAULT_MODEL,
    api_key=judge_api_key or api_key,
    temperature=0.0,
    max_tokens=1024,
)
st.session_state["judge_config"] = judge_config

# ── Sidebar: Caching Toggle ────────────────────────────────────────────────

st.sidebar.divider()
st.session_state["use_cache"] = st.sidebar.toggle(
    "Response caching", value=True, help="Cache identical requests to save cost"
)

# ── Run selected page ───────────────────────────────────────────────────────

pg.run()
