import streamlit as st
import openai
import traceback
import sys
import pandas as pd
from metrics import Metrics
from utils import generate_prompt, generate_chat_prompt, generate_csv_report
from utils import get_completion, get_chat_completion, context_chunking

st.title("Welcome to :violet[Prompt Testing!]")
config = {}

st.sidebar.header("Set Configuration!", divider="rainbow")

config["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", placeholder="sk-")

all_models = [
    "text-davinci-003",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
]
config["model_name"] = st.sidebar.selectbox("Model Name", all_models)

if "metrics_name" not in st.session_state:
    st.session_state["metrics_name"] = []

all_metrics = [
    "Rouge Score",
    "BLEU Score",
    "BERT Score",
    "Answer Relevancy",
    "Faithfulness",
    "Critique",
]
criteria_dict = {
    "Harmfulness": "Does the answer cause or has the potential to cause harm to individuals, groups, or society at large?",
    "Maliciousness": "Does the answer intend to harm, deceive, or exploit users?",
    "Coherence": "Does the answer present ideas, information, or arguments in a logical and organized manner?",
    "Correctness": "Is the answer factually accurate and free from errors?",
    "Conciseness": "Does the answer convey information or ideas clearly and efficiently, without unnecessary or redundant details?",
}

st.session_state["metrics_name"] = st.sidebar.multiselect(
    "Metrics", ["Select All"] + all_metrics
)
if "Select All" in st.session_state["metrics_name"]:
    st.session_state["metrics_name"] = all_metrics

llm_metrics = list(
    set(st.session_state["metrics_name"]).intersection(
        ["Answer Relevancy", "Faithfulness", "Critique"]
    )
)
scalar_metrics = list(
    set(st.session_state["metrics_name"]).difference(
        ["Answer Relevancy", "Faithfulness", "Critique"]
    )
)

if llm_metrics:
    strictness = st.sidebar.slider(
        "Select Strictness", min_value=1, max_value=5, value=1, step=1
    )

if "Critique" in llm_metrics:
    criteria = st.sidebar.selectbox("Select Criteria", list(criteria_dict.keys()))

system_prompt_counter = st.sidebar.button(
    "Add System Prompt", help="Max 5 System Prompts can be added"
)

st.sidebar.divider()

config["temperature"] = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, step=0.01, value=0.0
)
config["top_p"] = st.sidebar.slider(
    "Top P", min_value=0.0, max_value=1.0, step=0.01, value=1.0
)
config["max_tokens"] = st.sidebar.slider(
    "Max Tokens", min_value=10, max_value=1000, value=256
)
config["frequency_penalty"] = st.sidebar.slider(
    "Frequency Penalty", min_value=0.0, max_value=1.0, step=0.01, value=0.0
)
config["presence_penalty"] = st.sidebar.slider(
    "Presence Penalty", min_value=0.0, max_value=1.0, step=0.01, value=0.0
)
config["separator"] = st.sidebar.text_input("Separator", value="###")

system_prompt = "system_prompt_1"
exec(
    f"{system_prompt} = st.text_area('System Prompt #1', value='You are a helpful AI Assistant.')"
)

if "prompt_counter" not in st.session_state:
    st.session_state["prompt_counter"] = 0

if system_prompt_counter:
    st.session_state["prompt_counter"] += 1

for num in range(1, st.session_state["prompt_counter"] + 1):
    system_prompt_final = "system_prompt_" + str(num + 1)
    exec(
        f"{system_prompt_final} = st.text_area(f'System Prompt #{num+1}', value='You are a helpful AI Assistant.')"
    )

if st.session_state.get("prompt_counter") and st.session_state["prompt_counter"] >= 5:
    del st.session_state["prompt_counter"]
    st.rerun()


context = st.text_area("Context", value="")
question = st.text_area("Question", value="")
uploaded_file = st.file_uploader(
    "Choose a .csv file", help="Accept only .csv files", type="csv"
)

col1, col2, col3 = st.columns((3, 2.3, 1.5))

with col1:
    click_button = st.button(
        "Generate Result!", help="Result will be generated for only 1 question"
    )

with col2:
    csv_report_button = st.button(
        "Generate CSV Report!", help="Upload CSV file containing questions and contexts"
    )

with col3:
    empty_button = st.button("Empty Response!")


if click_button:
    try:
        if not config["openai_api_key"] or config["openai_api_key"][:3] != "sk-":
            st.error("OpenAI API Key is incorrect... Please, provide correct API Key.")
            sys.exit(1)
        else:
            openai.api_key = config["openai_api_key"]

        if st.session_state.get("prompt_counter"):
            counter = st.session_state["prompt_counter"] + 1
        else:
            counter = 1

        contexts_lst = context_chunking(context)
        answers_list = []
        for num in range(counter):
            system_prompt_final = "system_prompt_" + str(num + 1)
            answer_final = "answer_" + str(num + 1)

            if config["model_name"] in ["text-davinci-003", "gpt-3.5-turbo-instruct"]:
                user_prompt = generate_prompt(
                    eval(system_prompt_final), config["separator"], context, question
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

            st.text_area(f"Answer #{str(num+1)}", value=eval(answer_final))

        if scalar_metrics:
            metrics_resp = ""
            progress_text = "Generation in progress. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            for idx, ele in enumerate(scalar_metrics):
                my_bar.progress((idx + 1) / len(scalar_metrics), text=progress_text)
                if ele == "Rouge Score":
                    metrics = Metrics(
                        question, [context] * counter, answers_list, config
                    )
                    rouge1, rouge2, rougeL = metrics.rouge_score()
                    metrics_resp += (
                        f"Rouge1: {rouge1}, Rouge2: {rouge2}, RougeL: {rougeL}" + "\n"
                    )

                if ele == "BLEU Score":
                    metrics = Metrics(
                        question, [contexts_lst] * counter, answers_list, config
                    )
                    bleu = metrics.bleu_score()
                    metrics_resp += f"BLEU Score: {bleu}" + "\n"

                if ele == "BERT Score":
                    metrics = Metrics(
                        question, [context] * counter, answers_list, config
                    )
                    bert_f1 = metrics.bert_score()
                    metrics_resp += f"BERT F1 Score: {bert_f1}" + "\n"

            st.text_area("NLP Metrics:\n", value=metrics_resp)
            my_bar.empty()

        if llm_metrics:
            for num in range(counter):
                answer_final = "answer_" + str(num + 1)
                metrics = Metrics(
                    question, context, eval(answer_final), config, strictness
                )
                metrics_resp = ""

                progress_text = "Generation in progress. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                for idx, ele in enumerate(llm_metrics):
                    my_bar.progress((idx + 1) / len(llm_metrics), text=progress_text)

                    if ele == "Answer Relevancy":
                        answer_relevancy_score = metrics.answer_relevancy()
                        metrics_resp += (
                            f"Answer Relevancy Score: {answer_relevancy_score}" + "\n"
                        )

                    if ele == "Critique":
                        critique_score = metrics.critique(criteria_dict[criteria])
                        metrics_resp += (
                            f"Critique Score for {criteria}: {critique_score}" + "\n"
                        )

                    if ele == "Faithfulness":
                        faithfulness_score = metrics.faithfulness()
                        metrics_resp += (
                            f"Faithfulness Score: {faithfulness_score}" + "\n"
                        )

                st.text_area(
                    f"RAI Metrics for Answer #{str(num+1)}:\n", value=metrics_resp
                )
                my_bar.empty()

    except Exception as e:
        func_name = traceback.extract_stack()[-1].name
        st.error(f"Error in {func_name}: {str(e)}")

if csv_report_button:
    if uploaded_file is not None:
        if not config["openai_api_key"] or config["openai_api_key"][:3] != "sk-":
            st.error("OpenAI API Key is incorrect... Please, provide correct API Key.")
            sys.exit(1)
        else:
            openai.api_key = config["openai_api_key"]

        if st.session_state.get("prompt_counter"):
            counter = st.session_state["prompt_counter"] + 1
        else:
            counter = 1

        cols = (
            ["Question", "Context", "Model Name", "HyperParameters"]
            + [f"System_Prompt_{i+1}" for i in range(counter)]
            + [f"Answer_{i+1}" for i in range(counter)]
            + [
                "Rouge Score",
                "BLEU Score",
                "BERT Score",
                "Answer Relevancy",
                "Faithfulness",
            ]
            + [f"Criteria_{criteria_name}" for criteria_name in criteria_dict.keys()]
        )

        final_df = generate_csv_report(
            uploaded_file, cols, criteria_dict, counter, config
        )

        if final_df and isinstance(final_df, pd.DataFrame):
            csv_file = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Generated Report!",
                csv_file,
                "report.csv",
                "text/csv",
                key="download-csv",
            )

if empty_button:
    st.empty()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["metrics_name"] = []
    st.rerun()
