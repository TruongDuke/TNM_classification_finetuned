import os
import logging
import sys
import pandas as pd
import pickle
import re
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load API key from .env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

def get_csv(input_path):
    df = pd.read_csv(input_path)
    input_dic = []
    for i in range(len(df)):
        tmp_dic = {}
        tmp_dic["ID"] = df.loc[i, "ID"]
        tmp_dic["T"] = df.loc[i, "T"]
        tmp_dic["N"] = df.loc[i, "N"]
        tmp_dic["M"] = df.loc[i, "M"]
        with open(f"dataset/{df.loc[i, 'ID']}.txt", "r") as f:
            tmp_dic["Report"] = f.read()
        input_dic.append(tmp_dic)
    return input_dic

def process_tnm(process_dic, prefix="", model_name="gpt-3.5-turbo"):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    llm = ChatOpenAI(temperature=0, model_name=model_name)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)

    with open("init_data/tnm_en.txt", "r") as f:
        docu_text = f.read()

    first_input = (
        "You are an experienced thoracic surgeon. "
        "Based on the following TNM definitions, classify the TNM stage of a given radiology report. "
        "TNM Definitions: " + docu_text +
        "\nIf the report does not mention a certain finding, assume that it is normal."
    )
    conversation.predict(input=first_input)
    memory_init = pickle.dumps(conversation.memory)

    classification_prompt = PromptTemplate(
        input_variables=["report"],
        template="Understand the following radiology report and output its TNM classification. "
                 "Leave blank if there is no applicable information. "
                 "Please output **only** in the exact format: | T:value,N:value,M:value | "
                 "No explanation or extra information should be included.\n"
                 "Report:{report}"
    )
    classification_chain = LLMChain(llm=llm, prompt=classification_prompt)
    result_list = []

    for tmp_dic in process_dic:
        conv = ConversationChain(llm=llm, memory=pickle.loads(memory_init))
        input_str = (
            "Understand the following radiology report and output its TNM classification. "
            "Leave blank if there is no applicable information. "
            "Please output only in the exact format: {T:value,N:value,M:value}. "
            "No explanation or additional details are required.\n"
            "Report:" + tmp_dic["Report"]
        )
        result_str_1 = conv.predict(input=input_str)
        result_str_2 = classification_chain.run(result_str_1)

        tmp_dic_result = tmp_dic.copy()
        tmp_dic_result["result"] = result_str_1
        tmp_dic_result["classification"] = result_str_2
        tmp_dic_result["error_str"] = ""

        pattern = r"\|\s*(T\S*)\s*,\s*(N\S*)\s*,\s*(M\S*)\s*\|"
        match = re.search(pattern, result_str_2)
        try:
            tmp_dic_result[f"{prefix}_t"] = match.group(1)
        except:
            tmp_dic_result[f"{prefix}_t"] = "error"
            tmp_dic_result["error_str"] = result_str_2
        try:
            tmp_dic_result[f"{prefix}_n"] = match.group(2)
        except:
            tmp_dic_result[f"{prefix}_n"] = "error"
            tmp_dic_result["error_str"] = result_str_2
        try:
            tmp_dic_result[f"{prefix}_m"] = match.group(3)
        except:
            tmp_dic_result[f"{prefix}_m"] = "error"
            tmp_dic_result["error_str"] = result_str_2

        print(tmp_dic_result[f"{prefix}_t"], tmp_dic_result[f"{prefix}_n"], tmp_dic_result[f"{prefix}_m"])
        result_list.append(tmp_dic_result)

    return result_list

if __name__ == "__main__":
    input_dic = get_csv("dataset/test/label.csv")
    fine_tuned_model_id = "ft:gpt-3.5-turbo-0125:hust:tnm:BZYUeOiw"  # model fine-tuned
    result = process_tnm(input_dic, prefix="gpt-3.5-turbo", model_name=fine_tuned_model_id)
    df_result = pd.DataFrame(result)
    df_result.to_excel("test_out_result.xlsx", index=False)

