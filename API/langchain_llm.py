from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter,SpacyTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import re
import yaml
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
# from langchain.vectorstores import Pinecone as PineconeIndex
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import time


with open('config.yaml','r') as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI']['api_key']
model = config['OPENAI']['embade_model']
pinecone_api_key = config['pinecone']['api_key']
pinecone_env =  config['pinecone']['env']
directory = r"C:\Users\Programming.com\Desktop\MY_project\langchain_pdf_QA\langchain_pdf_QA\inputs"

llm = OpenAI(api_key=openai_api_key,temperature=0.9)
embeddings = OpenAIEmbeddings(api_key=openai_api_key,model="text-embedding-3-large")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "testpinecone"

def read_doc(directory):
    file_load = PyPDFDirectoryLoader(directory)
    document = file_load.load()
    return document


def clean_text(text):
    # Remove unnecessary whitespace and newlines
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    # Remove special characters (you can adjust this as needed)
    text = re.sub(r'[^\w\s\.\,\%\-\(\)]', '', text)  # Keep letters, numbers, punctuation, etc.
    
    # Normalize spacing again after removal
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def preprocess_data(doc):
    input_content = ""
    # Iterate over each Document and clean the page_content
    for document in doc:

        cleaned_content = clean_text(document.page_content)
        input_content += cleaned_content  # Update the document with cleaned content
    return input_content

def creating_chunks(input_content,chunk_size=500,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(input_content)
    return chunks

def creating_pinecone_index(chunks):

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore.from_texts(chunks, embeddings, index_name=index_name)
    return vector_store


def retriving_answer(query,vector_store):

    chain = load_qa_chain(llm=llm,chain_type="refine") #map_rerank,stuff,refine
    docs = vector_store.similarity_search(query)
    responce = chain.run(input_documents=docs, question=query)
    return responce

