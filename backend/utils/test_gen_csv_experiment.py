import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain.docstore.document import Document
import json


# Generates synthetic test cases using GPT-4o, saves them to a CSV file testset_syntethic.csv
# Load CSV data
df = pd.read_csv("backend/data/ferry_trips_data.csv")

# Load semantic model
with open("backend/data/semantic_model.json", "r") as f:
    semantic_model = json.load(f)

# Create LangChain documents
docs = []
for index, row in df.iterrows():
    content = ", ".join([f"{col}: {val}" for col, val in row.items()])
    metadata = {"row_index": index, "route_id": row["route_id"],"ferry_name": row["ferry_name"]} #add metadata
    docs.append(Document(page_content=content, metadata=metadata))

# Prepare LLM and embeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Generate Testset
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
test_pd = dataset.to_pandas()
test_pd.to_csv("../data/ragas/testset_syntethic.csv")