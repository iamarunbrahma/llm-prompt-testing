# LLM - Prompt Testing
## Objective:
As LLM developers, we often face challenges in fine-tuning prompts to generate model answer which is more aligned with ground truth answer. Hence, I created this framework so that anyone can run this streamlit app to add multiple system prompts, fine-tune each prompt (using chain-of-thought, few-shot etc.), and then compare multiple system prompts based on the model-generated answer quality. Quality of answers can be measured using NLP metrics such as ROUGE, BLEU, or BERTScore and Responsible AI metrics such as Faithfulness, Answer Relevancy Score, Harmfulness etc.

## Natural Language Processing (NLP) Metrics:
* ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
* BLEU
* BERTScore ('distilbert-base-uncased' model is being used to compute BERTScore).

## Responsible AI (RAI) Metrics:
* Answer Relevancy Score: Regenerate the question from the model-generated answer and compute a cosine similarity score between the actual question and the regenerated question. If the similarity score is high, it implies that the answer is relevant to the actual question.
* Harmfulness: Check if the model-generated answer is potentially harmful to individuals, groups, or society at large.
* Maliciousness: Check if the model-generated answer intends to harm, deceive, or exploit users.
* Coherence: Check if the model-generated answer represents information or arguments in a logical and organized manner.
* Correctness: Check if the model-generated answer is factually accurate and free from errors.
* Conciseness: Check if the model-generated answer conveys factual information clearly and efficiently, without unnecessary or redundant details.
* Faithfulness: Generate multiple factual statements from model-generated response and question. Given the context and factual statements, determine whether these statements are supported by the information present in the context. If these statements entail the given context, the final verdict should be yes or No.


## Configuration Settings:
* Model Name: Select a model to generate the answer
* Strictness: Send the same final concatenated prompt to the LLM model multiple times and take the majority result as the final answer for each RAI metric.
* Add System Prompt: Define multiple system prompts to generate multiple answers for each question.
* Separator: Delimiter to separate system prompt, context and question in the final concatenated prompt.

## Generate CSV Report:
Upload a CSV file having Questions and Contexts. Write multiple prompts and change hyperparameters. Click on "Generate CSV Report" to generate all the metric results for each question and it's corresponding context.

## How to run locally:
If you want to run this app locally, first clone this repo using `git clone`.<br><br>
Now, install all libraries by running the following command in the terminal:<br>
```python
pip install -r requirements.txt
```
  
Now, run the app from the terminal:  
```python
streamlit run app.py
```

Provide your own OpenAI API Key to generate answers and metrics. 

This project is hosted on HuggingFace spaces: [Live Demo of LLM - Prompt Testing](https://huggingface.co/spaces/heliosbrahma/llm-prompt-testing).<br><br>
_If you have any queries, you can open an issue. If you like this project, please ⭐ this repository._
