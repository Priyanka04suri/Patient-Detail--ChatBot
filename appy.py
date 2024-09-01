import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

with open('data.json', 'r') as file:
    json_data = json.load(file)

def person_to_text(person, indent=0):
    text_data = ""
    for key, value in person.items():
        if isinstance(value, dict):
            text_data += '  ' * indent + f"{key}:\n"
            text_data += person_to_text(value, indent + 1)
        elif isinstance(value, list):
            text_data += '  ' * indent + f"{key}:\n"
            for item in value:
                if isinstance(item, dict):
                    text_data += person_to_text(item, indent + 1)
                else:
                    text_data += '  ' * (indent + 1) + f"- {item}\n"
        else:
            text_data += '  ' * indent + f"{key}: {value}\n"
    return text_data

def json_to_text(data):
    all_persons_text = ""
    for person in data:
        all_persons_text += "Person Data:\n"
        all_persons_text += person_to_text(person)
        all_persons_text += "\n" + "-"*30 + "\n"  
    return all_persons_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Patient Detail")

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)
        text_format = json_to_text(json_data)
        text_chunks = get_text_chunks(text_format)
        get_vector_store(text_chunks)
        st.success("Done")

if __name__ == "__main__":
    main()
       