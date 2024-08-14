import openai
import langchain_community
import pinecone 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
# from openai import OpenAI
import yaml
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.openai import OpenAI



# pc = Pinecone(api_key="********-****-****-****-************")

with open('config.yaml','r') as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI']['api_key']
model = config['OPENAI']['model']
pinecone_api_key = config['pinecone']['api_key']
pinecone_env =  config['pinecone']['env']
directory = r"C:\Users\Programming.com\Desktop\MY_project\langchain_pdf_QA\langchain_pdf_QA\inputs"

Client = OpenAIEmbeddings(api_key=openai_api_key)
llm_open_ai_client = OpenAI(model_name=model,temperature=0.5)

def read_doc(directory):
    file_load = PyPDFDirectoryLoader(directory)
    document = file_load.load()
    return document


# creating documents in to chunks

def creating_chunks(document,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(document)
    return chunks


def pinecone_config(query,k=2):
    doc = read_doc(directory)
    pinecone.init(
        api_key = pinecone_api_key,
        environment = pinecone_env
    )
    index_name = "langchainvector"

    index=Pinecone.from_documents(doc,Client,index_name=index_name)
    machin_result = index.similarity_search(query,k)
    return machin_result


def retriving_answer(query):

    chain = load_qa_chain(llm_open_ai_client,chain_type='stuff')
    doc_search = pinecone_config(query,k=2)
    print(doc_search)
    responce = chain.run(input_documents=doc_search,question=query)
    return responce
