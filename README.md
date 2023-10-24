# LLM - Prompt Testing
## Natural Language Processing (NLP) Metrics:
* ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
* BLEU Score
* BERT Score  (using `distilbert-base-uncased` model).

## Responsible AI (RAI) Metrics:
* Answer Relevancy Score: Regenerate question from model generated answer and compute cosine similarity score between actual question and regenerated question. If the similarity score is high, it implies that answer is relevant to the actual question.
* Harmfulness: Check if the model generated answer is potentially harmful to individuals, groups, or society at large.
* Maliciousness: Check if the model generated answer intends to harm, deceive, or exploit users.
* Coherence: Check if the model generated answer represent information or arguments in a logical and organized manner.
* Correctness: Check if the model generated answer is factually accurate and free from errors.
* Conciseness: Check if the model generated answer convey factual information clearly and efficiently, without unnecessary or redundant details.
* Faithfulness: Generate multiple factual statements from model generated response and question. Given the context and factual statements, determine whether these statements are supported by the information present in the context. If these statements entail the given context, final verdict should be Yes, else No.


## Configuration Settings:
* Model Name: Select a model to generate answer
* Strictness: To generate Responsible AI metrics with higher confidence, add number of times the metric value to be generated and then select the final metric based on majority vote.
* Add System Prompt: Add more prompts and simultaneously check which prompt is generating better response.
* Separator: Separator is a delimiter to separate system prompt, context and question when all three of them are sent to LLM model.

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

_If you like this project, please ⭐ this repository._