# LLM Prompt Testing Framework

A Streamlit app to test and compare LLM system prompts. Write multiple prompts, generate answers from any LLM provider, and measure quality using NLP and LLM-based metrics.

## Features

**Multi-provider support** via [LiteLLM](https://github.com/BerriAI/litellm):
- OpenAI (GPT-4o, o4-mini, o3-mini)
- Anthropic (Claude Sonnet, Opus, Haiku)
- Google (Gemini 2.5 Pro, Flash)
- Ollama (Llama 3, Mistral, local models)
- 100+ others with custom model names

**NLP Metrics** (compared against a ground truth reference answer):
- ROUGE (1, 2, L), BLEU, BERTScore

**LLM Judge Metrics** (a separate model scores the answers):
- Answer Relevancy: generates a question from the answer and checks cosine similarity to the original
- Faithfulness: extracts factual statements and verifies them against the context (returns a 0-1 ratio)
- Critique: binary yes/no on criteria like harmfulness, coherence, correctness
- Rubric Scoring: user-defined 1-5 scale criteria
- Pairwise Comparison: head-to-head with position debiasing (runs both orderings)

**Other capabilities:**
- Compare up to 10 system prompts side by side
- Prompt templates with `{{variable}}` placeholders
- Response caching to skip redundant API calls
- Token count, latency, and cost tracking per request
- Batch evaluation from CSV files
- Separate judge model config (use a cheaper model for scoring)
- Comparison dashboard with charts and JSON/CSV export

## Pages

| Page | What it does |
|------|-------------|
| Prompt Lab | Test one question with multiple prompts, see metrics |
| Batch Eval | Upload a CSV, evaluate all rows, download results |
| Comparison | Charts, pairwise matrix, cost summary, export |

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Cloud providers** (OpenAI, Anthropic, Google): paste your API key in the sidebar.

**Ollama** (local): install [Ollama](https://ollama.ai), run `ollama pull llama3`, select "ollama" as the provider. No API key needed.

**Custom providers**: toggle "Custom model name" and enter the LiteLLM model ID (e.g. `together_ai/meta-llama/Llama-3-70b`).

## CSV Format

For batch evaluation, your CSV needs question and context columns. A ground truth column is optional but required for NLP metrics.

| Question | Context | Ground Truth |
|----------|---------|-------------|
| What is X? | X is defined as... | X is a concept that... |

Column names are auto-detected. You can remap them manually if needed.

## Project Structure

```
app.py                  Entry point, sidebar config, navigation
pages/
  1_prompt_lab.py       Single-question testing + metrics
  2_batch_eval.py       CSV batch processing
  3_comparison.py       Results visualization + export
core/
  schemas.py            Pydantic data models
  llm_client.py         LiteLLM wrapper, caching, cost tracking
  metrics.py            NLP metrics + LLM judge evaluation
  cache.py              Hash-based response caching
  templates.py          Template variable rendering
```

## License

MIT. See [LICENSE](LICENSE).
