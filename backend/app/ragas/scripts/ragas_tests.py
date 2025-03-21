import pandas as pd
import json
import requests
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, SemanticSimilarity, BleuScore, RougeScore
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import datetime
from app.ragas.custom_metrics.LenientFactualCorrectness import LenientFactualCorrectness

load_dotenv()
# This is the main RAGAS Evaluation script. It is used to evaluate the performance of the agent.
API_URL = os.getenv('API_URL')
RAGAS_APP_TOKEN = os.getenv('RAGAS_APP_TOKEN')

# Initialize using Hugging Face models
evaluator_llm = LangchainLLMWrapper(
    HuggingFaceEndpoint(
        repo_id="google/flan-t5-xl",  # Use a smaller model that's still effective
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

def run_test_case(query, ground_truth=None):
    api_url = f"{API_URL}/api/query"
    try:
        response = requests.post(api_url, json={
            "question": query,
            "source_file": "ferry_trips_data.csv"
        })
        response.raise_for_status()
        
        response_data = response.json()
        agent_response = response_data.get('content')  # This is already the clean response
        full_response = response_data.get('full_response')  # This contains the full context
        sql_queries = response_data.get('sql_queries', [])
        
        if agent_response is None:
            print(f"Error: No 'content' key found in the API response for query: {query}")
            return None, None, False
            
        # Format the complete context with reasoning and SQL
        contexts = []
        for sql in sql_queries:
            contexts.append(f"SQL Query: {sql}")
        
        # Add the complete agent response as context
        contexts.append(f"Agent Reasoning and Response: {full_response}")
        
        # This is the original return format: response, context, success
        return agent_response, contexts, True
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for query: {query}: {e}")
        return None, None, False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for query: {query}: {e}")
        return None, None, False

# Load test cases from JSON file
test_cases_path = Path("app/ragas/test_cases/test_cases.json")
try:
    with open(test_cases_path, "r") as f:
        test_cases_list = json.load(f)
except FileNotFoundError:
    print(f"Error: {test_cases_path} not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {test_cases_path}.")
    exit()

results = []

for test_case in test_cases_list:
    query = test_case['user_input']
    ground_truth = test_case.get('reference')
    response, context, api_call_success = run_test_case(query, ground_truth)
    results.append({
        "user_input": query,
        "reference": ground_truth,
        "response": response,
        "context": context,
        "api_call_success": api_call_success
    })

results_df = pd.DataFrame(results)

# RAGAS Evaluation
ragas_data = pd.DataFrame({
    "user_input": results_df['user_input'],
    "reference": results_df['reference'],
    "response": results_df['response'],
    "retrieved_contexts": results_df['context'].apply(lambda x: x if isinstance(x, list) else [])
})

# Create EvaluationDataset
eval_dataset = EvaluationDataset.from_pandas(ragas_data)

# Use a mix of LLM-dependent and non-LLM metrics
metrics = [
    LenientFactualCorrectness(),
    SemanticSimilarity(embeddings=evaluator_embeddings),
    LLMContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    BleuScore(),
    RougeScore()
]

print(ragas_data[['user_input', 'response', 'reference']])

# Run evaluation
ragas_results = evaluate(
    eval_dataset,
    metrics=metrics
)

# Also comment out the upload line (line 112)
# ragas_results.upload()

# Add RAGAS metrics to results_df
for metric_name, scores in ragas_results.to_pandas().items():
    if metric_name != 'hash':
        results_df[metric_name] = scores

# Save results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"output/ragas_results_{timestamp}.csv", index=False)

cwd = Path(__file__).parent.parent.parent.parent.resolve()  # Go up to root directory
output_dir = cwd.joinpath("output")  # Use the same output directory as app.py

# Create timestamped directory within output
timestamped_dir = output_dir.joinpath(f"ragas_{timestamp}")
timestamped_dir.mkdir(exist_ok=True)

# Save results using timestamped paths
results_df.to_csv(timestamped_dir.joinpath("test_results.csv"), index=False)
print(f"Test results saved to {timestamped_dir}/test_results.csv")

# Save metrics summary
metrics_df = ragas_results.to_pandas()
metrics_df.to_csv(timestamped_dir.joinpath("metrics_summary.csv"), index=False)

print(f"Results saved in directory: {timestamped_dir}")
print("\nRAGAS Results:")
print(ragas_results)