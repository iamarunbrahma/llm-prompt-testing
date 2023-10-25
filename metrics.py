from collections import Counter
import evaluate
import streamlit as st
import traceback
import numpy as np
from numpy.linalg import norm
from utils import get_embeddings, get_chat_completion


class Metrics:
    def __init__(self, question, context, answer, config, strictness=1):
        self.question = question
        self.context = context
        self.answer = answer
        self.strictness = strictness

        config["model_name"] = "gpt-3.5-turbo"
        self.config = config

    def rouge_score(self):
        try:
            if not self.answer or not self.context:
                raise ValueError(
                    "Please provide both context and answer to generate Rouge Score."
                )

            rouge = evaluate.load("rouge")
            results = rouge.compute(predictions=self.answer, references=self.context)
            rouge1 = np.round(results["rouge1"], 3)
            rouge2 = np.round(results["rouge2"], 3)
            rougeL = np.round(results["rougeL"], 3)
            return rouge1, rouge2, rougeL

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")

    def bleu_score(self):
        try:
            if not self.answer or not self.context:
                raise ValueError(
                    "Please provide both context and answer to generate BLEU Score."
                )

            bleu = evaluate.load("bleu")
            results = bleu.compute(predictions=self.answer, references=self.context)
            return np.round(results["bleu"], 3)

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")

    def bert_score(self):
        try:
            if not self.answer or not self.context:
                raise ValueError(
                    "Please provide both context and answer to generate BLEU Score."
                )

            bertscore = evaluate.load("bertscore")
            results = bertscore.compute(
                predictions=self.answer,
                references=self.context,
                lang="en",
                model_type="distilbert-base-uncased",
            )
            return np.round(results["f1"], 3)

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")

    def answer_relevancy(self):
        try:
            if not self.answer or not self.question:
                raise ValueError(
                    "Please provide both question and answer to generate Answer Relevancy Score."
                )

            relevancy_prompt = """
            Generate question for the given answer.

            Here are few examples:
            Answer: The first ODI Cricket World Cup was held in 1975, and the West Indies cricket team won the tournament. Clive Lloyd was the captain of the winning West Indies team. They defeated Australia in the final to become the first-ever ODI Cricket World Cup champions.
            Question: Which team won the first ODI Cricket World Cup and in which year? Who was the captain of the winning team?

            Answer: The first president of the United States of America was George Washington. He became president in the year 1789. Washington served as the country's first president from April 30, 1789, to March 4, 1797.
            Question: Who was the first president of the United States of America and in which year did he become president?
            
            Using the answer provided below, generate a question which is relevant to the answer.
            """

            answer_relevancy_score = []

            for _ in range(self.strictness):
                generated_question = get_chat_completion(
                    self.config, relevancy_prompt, self.answer
                )
                question_vec = np.asarray(get_embeddings(self.question.strip()))
                generated_question_vec = np.asarray(
                    get_embeddings(generated_question.strip())
                )
                score = np.dot(generated_question_vec, question_vec) / (
                    norm(generated_question_vec) * norm(question_vec)
                )
                answer_relevancy_score.append(score)

            return np.round(np.mean(answer_relevancy_score), 3)

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")

    def critique(self, criteria):
        try:
            if not self.answer or not self.question:
                raise ValueError(
                    "Please provide both question and answer to generate Critique Score."
                )

            critique_prompt = """
            Given a question and answer. Evaluate the answer only using the given criteria. 
            Think step by step providing reasoning and arrive at a conclusion at the end by generating a Yes or No verdict at the end.
                        
            Here are few examples:
            question: Who was the president of the United States of America when World War 2 happened?
            answer: Franklin D. Roosevelt was the President of the United States when World War II happened. He served as President from 1933 until his death in 1945, which covered the majority of the war years.
            criteria: Is the output written in perfect grammar
            Here are my thoughts: the criteria for evaluation is whether the output is written in perfect grammar. In this case, the output is grammatically correct. Therefore, the answer is:\n\nYes
            """

            responses = []
            answer_dict = {"Yes": 1, "No": 0}
            reversed_answer_dict = {1: "Yes", 0: "No"}
            critique_input = f"question: {self.question}\nanswer: {self.answer}\ncriteria: {criteria}\nHere are my thoughts:"

            for _ in range(self.strictness):
                response = get_chat_completion(
                    self.config, critique_prompt, critique_input
                )
                response = response.split("\n\n")[-1]
                responses.append(response)

            if self.strictness > 1:
                critique_score = Counter(
                    [answer_dict.get(response, 0) for response in responses]
                ).most_common(1)[0][0]
            else:
                critique_score = answer_dict.get(responses[-1], 0)

            return reversed_answer_dict[critique_score]

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")

    def faithfulness(self):
        try:
            if not self.answer or not self.question or not self.context:
                raise ValueError(
                    "Please provide context, question and answer to generate Faithfulness Score."
                )

            generate_statements_prompt = """
            Given a question and answer, create one or more statements from each sentence in the given answer.
            question: Who is Sachin Tendulkar and what is he best known for?
            answer: Sachin Tendulkar is a former Indian cricketer widely regarded as one of the greatest batsmen in the history of cricket. He is often referred to as the "Little Master" or the "Master Blaster" and is considered a cricketing legend.
            statements:\nSachin Tendulkar is a former Indian cricketer.\nSachin Tendulkar is widely regarded as one of the greatest batsmen in the history of cricket.\nHe is often referred to as the "Little Master" or the "Master Blaster."\nSachin Tendulkar is considered a cricketing legend.
            question: What is the currency of Japan?
            answer: The currency of Japan is the Japanese Yen, abbreviated as JPY. 
            statements:\nThe currency of Japan is the Japanese Yen.\nThe Japanese Yen is abbreviated as JPY.
            question: Who was the president of the United States of America when World War 2 happened?
            answer: Franklin D. Roosevelt was the President of the United States when World War II happened. He served as President from 1933 until his death in 1945, which covered the majority of the war years.
            statements:\nFranklin D. Roosevelt was the President of the United States during World War II.\nFranklin D. Roosevelt served as President from 1933 until his death in 1945.
            """

            generate_statements_input = (
                f"question: {self.question}\nanswer: {self.answer}\nstatements:\n"
            )

            faithfulness_score = []

            for _ in range(self.strictness):
                generated_statements = get_chat_completion(
                    self.config, generate_statements_prompt, generate_statements_input
                )
                generated_statements = "\n".join(
                    [
                        f"{i+1}. {st}"
                        for i, st in enumerate(generated_statements.split("\n"))
                    ]
                )

                nli_prompt = """
                Prompt: Natural language inference
                Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

                Context:\nJames is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. James is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
                Statements:\n1. James is majoring in Biology.\n2. James is taking a course on Artificial Intelligence.\n3. James is a dedicated student.\n4. James has a part-time job.\n5. James is interested in computer programming.\n
                Answer:
                1. James is majoring in Biology.
                Explanation: James's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
                2. James is taking a course on Artificial Intelligence.
                Explanation: The context mentions the courses James is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that James is taking a course on AI. Verdict: No.
                3. James is a dedicated student.
                Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
                4. James has a part-time job.
                Explanation: There is no information given in the context about James having a part-time job. Therefore, it cannot be deduced that James has a part-time job.  Verdict: No.
                5. James is interested in computer programming.
                Explanation: The context states that James is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
                Final verdict for each statement in order: No. No. Yes. No. Yes.
                """

                nli_input = f"Context:\n{self.context}\nStatements:\n{generated_statements}\nAnswer:"

                results = get_chat_completion(self.config, nli_prompt, nli_input)
                results = results.lower().strip()

                final_answer = "Final verdict for each statement in order:".lower()
                if results.find(final_answer) != -1:
                    results = results[results.find(final_answer) + len(final_answer) :]
                    results_lst = [ans.lower().strip() for ans in results.split(".")]
                    score = max(results_lst).capitalize()

                else:
                    no_count = results.count("verdict: no")
                    yes_count = results.count("verdict: yes")
                    score = "Yes" if yes_count >= no_count else "No"

                faithfulness_score.append(score)

            return max(faithfulness_score)

        except Exception as e:
            func_name = traceback.extract_stack()[-1].name
            st.error(f"Error in {func_name}: {str(e)}")
