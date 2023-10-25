from collections import defaultdict
import traceback
import openai
from openai.error import OpenAIError
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
import streamlit as st
import pandas as pd


def generate_prompt(system_prompt, separator, context, question):
    user_prompt = ""

    if system_prompt:
        user_prompt += system_prompt + separator
    if context:
        user_prompt += context + separator
    if question:
        user_prompt += question + separator

    return user_prompt


def generate_chat_prompt(separator, context, question):
    user_prompt = ""

    if context:
        user_prompt += context + separator
    if question:
        user_prompt += question + separator

    return user_prompt


@retry(wait=wait_random_exponential(min=3, max=90), stop=stop_after_attempt(6))
def get_embeddings(text, embedding_model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=embedding_model,
        input=text,
    )
    embedding_vectors = response["data"][0]["embedding"]
    return embedding_vectors


@retry(wait=wait_random_exponential(min=3, max=90), stop=stop_after_attempt(6))
def get_completion(config, user_prompt):
    try:
        response = openai.Completion.create(
            model=config["model_name"],
            prompt=user_prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
        )

        answer = response["choices"][0]["text"]
        answer = answer.strip()
        return answer

    except OpenAIError as e:
        func_name = traceback.extract_stack()[-1].name
        st.error(f"Error in {func_name}:\n{type(e).__name__}=> {str(e)}")


@retry(wait=wait_random_exponential(min=3, max=90), stop=stop_after_attempt(6))
def get_chat_completion(config, system_prompt, question):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = openai.ChatCompletion.create(
            model=config["model_name"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
        )

        answer = response["choices"][0]["message"]["content"]
        answer = answer.strip()
        return answer

    except OpenAIError as e:
        func_name = traceback.extract_stack()[-1].name
        st.error(f"Error in {func_name}:\n{type(e).__name__}=> {str(e)}")


def context_chunking(context, threshold=512, chunk_overlap_limit=0):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    contexts_lst = []
    while len(encoding.encode(context)) > threshold:
        context_temp = encoding.decode(encoding.encode(context)[:threshold])
        contexts_lst.append(context_temp)
        context = encoding.decode(
            encoding.encode(context)[threshold - chunk_overlap_limit :]
        )

    if context:
        contexts_lst.append(context)

    return contexts_lst


def generate_csv_report(file, cols, criteria_dict, counter, config):
    try:
        df = pd.read_csv(file)

        if "Questions" not in df.columns or "Contexts" not in df.columns:
            raise ValueError(
                "Missing Column Names in .csv file: `Questions` and `Contexts`"
            )

        final_df = pd.DataFrame(columns=cols)
        hyperparameters = f"Temperature: {config['temperature']}\nTop P: {config['top_p']} \
        \nMax Tokens: {config['max_tokens']}\nFrequency Penalty: {config['frequency_penalty']} \
        \nPresence Penalty: {config['presence_penalty']}"

        progress_text = "Generation in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)

        for idx, row in df.iterrows():
            my_bar.progress((idx + 1) / len(df), text=progress_text)

            question = row["Questions"]
            context = row["Contexts"]
            contexts_lst = context_chunking(context)

            system_prompts_list = []
            answers_list = []
            for num in range(counter):
                system_prompt_final = "system_prompt_" + str(num + 1)
                system_prompts_list.append(eval(system_prompt_final))

                if config["model_name"] in [
                    "text-davinci-003",
                    "gpt-3.5-turbo-instruct",
                ]:
                    user_prompt = generate_prompt(
                        eval(system_prompt_final),
                        config["separator"],
                        context,
                        question,
                    )
                    exec(f"{answer_final} = get_completion(config, user_prompt)")

                else:
                    user_prompt = generate_chat_prompt(
                        config["separator"], context, question
                    )
                    exec(
                        f"{answer_final} = get_chat_completion(config, eval(system_prompt_final), user_prompt)"
                    )

                answers_list.append(eval(answer_final))

            from metrics import Metrics

            metrics = Metrics(question, [context] * counter, answers_list, config)
            rouge1, rouge2, rougeL = metrics.rouge_score()
            rouge_scores = f"Rouge1: {rouge1}, Rouge2: {rouge2}, RougeL: {rougeL}"

            metrics = Metrics(question, [contexts_lst] * counter, answers_list, config)
            bleu = metrics.bleu_score()
            bleu_scores = f"BLEU Score: {bleu}"

            metrics = Metrics(question, [context] * counter, answers_list, config)
            bert_f1 = metrics.bert_score()
            bert_scores = f"BERT F1 Score: {bert_f1}"

            answer_relevancy_scores = []
            critique_scores = defaultdict(list)
            faithfulness_scores = []
            for num in range(counter):
                answer_final = "answer_" + str(num + 1)
                metrics = Metrics(
                    question, context, eval(answer_final), config, strictness=3
                )

                answer_relevancy_score = metrics.answer_relevancy()
                answer_relevancy_scores.append(
                    f"Answer #{str(num+1)}: {answer_relevancy_score}"
                )

                for criteria_name, criteria_desc in criteria_dict.items():
                    critique_score = metrics.critique(criteria_desc, strictness=3)
                    critique_scores[criteria_name].append(
                        f"Answer #{str(num+1)}: {critique_score}"
                    )

                faithfulness_score = metrics.faithfulness(strictness=3)
                faithfulness_scores.append(
                    f"Answer #{str(num+1)}: {faithfulness_score}"
                )

            answer_relevancy_scores = ";\n".join(answer_relevancy_scores)
            faithfulness_scores = ";\n".join(faithfulness_scores)

            critique_scores_lst = []
            for criteria_name in criteria_dict.keys():
                score = ";\n".join(critique_scores[criteria_name])
                critique_scores_lst.append(score)

            final_df.loc[len(final_df)] = (
                [question, context, config["model_name"], hyperparameters]
                + system_prompts_list
                + answers_list
                + [
                    rouge_scores,
                    bleu_scores,
                    bert_scores,
                    answer_relevancy_score,
                    faithfulness_score,
                ]
                + critique_scores_lst
            )

        my_bar.empty()
        return final_df

    except Exception as e:
        func_name = traceback.extract_stack()[-1].name
        st.error(f"Error in {func_name}: {str(e)}, {traceback.format_exc()}")
