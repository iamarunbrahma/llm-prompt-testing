# LLM Prompt Testing Framework v2.0

A Streamlit-based framework for systematically testing and comparing LLM system prompts across multiple providers. Evaluate answer quality using NLP metrics and LLM-as-Judge evaluation.

## Features

### Multi-Provider Support
Test prompts across any LLM provider via [LiteLLM](https://github.com/BerriAI/litellm):
- **OpenAI**: GPT-4o, GPT-4 Turbo, o4-mini, o3-mini
- **Anthropic**: Claude Sonnet/Opus/Haiku
- **Google**: Gemini 2.5 Pro, Gemini Flash
- **Ollama**: Llama 3, Mistral, CodeLlama (local)
- **100+ other providers** via custom model names

### Evaluation Metrics

**NLP Metrics** (compare against ground truth reference):
- **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU**
- **BERTScore** (using `distilbert-base-uncased`)

**LLM Judge Metrics** (model-based evaluation):
- **Answer Relevancy** — Regenerate question from answer, measure cosine similarity to original
- **Faithfulness** — Extract factual statements, verify against context via NLI (returns 0.0-1.0 ratio)
- **Critique** — Binary evaluation against criteria (Harmfulness, Coherence, Correctness, etc.)
- **Rubric Scoring** — User-defined 1-5 scale criteria with custom descriptions
- **Pairwise Comparison** — Head-to-head comparison with reasoning

### Key Capabilities
- Compare up to 10 system prompts side-by-side
- **Prompt templates** with `{{variable}}` support for sweep testing
- **Response caching** to avoid redundant API calls
- **Cost & latency tracking** per request (tokens in/out, estimated cost)
- **Batch CSV evaluation** with column auto-mapping
- **Separate judge model** configuration (use a different model for evaluation)
- **Comparison dashboard** with charts, pairwise matrix, and export

## Pages

| Page | Description |
|------|-------------|
| **Prompt Lab** | Single-question testing with full metrics |
| **Batch Eval** | CSV upload for bulk evaluation |
| **Comparison** | Visualize and export results |

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

### Provider setup

**OpenAI / Anthropic / Google**: Enter your API key in the sidebar.

**Ollama (local)**: Install [Ollama](https://ollama.ai), pull a model (`ollama pull llama3`), and select "ollama" as the provider. No API key needed.

**Custom providers**: Toggle "Custom model name" in the sidebar and enter the LiteLLM model identifier (e.g., `together_ai/meta-llama/Llama-3-70b`).

## CSV Format

For batch evaluation, your CSV should have columns for questions and contexts. A ground truth column is optional but enables NLP metrics.

| Question | Context | Ground Truth |
|----------|---------|-------------|
| What is X? | X is defined as... | X is a concept that... |

Column names are auto-detected. You can manually map them if they differ.

## Architecture

```
app.py                  → Entry point, sidebar config, navigation
pages/
  1_prompt_lab.py       → Single-question testing + metrics
  2_batch_eval.py       → CSV batch processing
  3_comparison.py       → Results visualization + export
core/
  schemas.py            → Pydantic data models (immutable config)
  llm_client.py         → LiteLLM wrapper with caching + cost tracking
  metrics.py            → NLPMetrics + LLMJudge evaluation engine
  cache.py              → SHA-256 hash-based response caching
  templates.py          → {{variable}} template rendering
```

## License

MIT License - see [LICENSE](LICENSE) for details.
