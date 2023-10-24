# LLM - Prompt Testing
## Natural Language Processing (NLP) Metrics:
* ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
* BLEU Score
* BERT Score  (using `distilbert-base-uncased` model).

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
* Strictness: To generate Responsible AI metrics with higher confidence, add a number of times the metric value to be generated and then select the final metric based on majority vote.
* Add System Prompt: Add more prompts and simultaneously check which prompt is generating better response.
* Separator: A separator is a delimiter to separate system prompt, context and question when all three of them are sent to LLM model.

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

_If you like this project, please ⭐ this repository._
