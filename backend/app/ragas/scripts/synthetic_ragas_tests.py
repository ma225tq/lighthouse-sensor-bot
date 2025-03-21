import pandas as pd
import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, SemanticSimilarity
from ragas.metrics import RougeScore
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import datetime
import requests
from app.ragas.custom_metrics.LenientFactualCorrectness import LenientFactualCorrectness
from app.ragas.custom_metrics.bleu_score import BleuScore

load_dotenv()
# this script is used to evaluate the performance of the agent on the synthetic dataset.
API_URL = os.getenv('API_URL')
RAGAS_APP_TOKEN = os.getenv('RAGAS_APP_TOKEN')

# Initialize LLM and Embeddings wrappers
evaluator_llm = None
try:
    print("Initializing evaluator with gpt2 model")
    hf_llm = HuggingFaceEndpoint(
        repo_id="gpt2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
    # Test by actually generating text
    test_text = hf_llm.invoke("Hello world")
    print(f"Test output: {test_text[:20]}...")
    evaluator_llm = LangchainLLMWrapper(hf_llm)
except Exception as e:
    print(f"Failed to initialize LLM: {e}")
    evaluator_llm = None
    print("WARNING: Continuing with non-LLM metrics only")

evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

def load_synthetic_test_cases():
    """Load test cases from the synthetic test cases JSON file"""
    test_cases_path = Path("app/ragas/test_cases/synthetic_test_cases.json")
    try:
        with open(test_cases_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {test_cases_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {test_cases_path}.")
        return None

def run_test_case(query, ground_truth=None):
    api_url = f"{API_URL}/api/query"
    try:
        # Always include the source_file parameter with the query
        response = requests.post(api_url, json={
            "question": query,
            "source_file": "ferry_trips_data.csv"  # Add this to prevent 400 Bad Request
        })
        response.raise_for_status()
        
        response_data = response.json()
        agent_response = response_data.get('content')
        
        if agent_response is None:
            # Return an empty string instead of None to prevent RAGAS errors
            return "", [], False
            
        # Extract queries and context if available
        sql_queries = response_data.get('sql_queries', [])
        full_response = response_data.get('full_response', '')
        
        # Format the complete context with reasoning and SQL
        contexts = []
        for sql in sql_queries:
            contexts.append(f"SQL Query: {sql}")
        
        # Add the complete agent response as context
        if full_response:
            contexts.append(f"Agent Reasoning and Response: {full_response}")
        
        return agent_response, contexts, True
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for query: {query}: {e}")
        return "", [], False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for query: {query}: {e}")
        return "", [], False

def run_synthetic_evaluation():
    """Run evaluation using the synthetic test cases"""
    # Load synthetic test cases
    test_cases = load_synthetic_test_cases()
    if test_cases is None:
        return
    
    results = []
    
    # Process each test case
    for test_case in test_cases:
        query = test_case['user_input']
        ground_truth = test_case['reference']
        response, context, api_call_success = run_test_case(query, ground_truth)
        
        results.append({
            "user_input": query,
            "reference": ground_truth,
            "response": response,
            "context": context,
            "reference_contexts": test_case['reference_contexts'],
            "api_call_success": api_call_success
        })
    
    # Create results DataFrame for RAGAS evaluation
    results_df = pd.DataFrame(results)
    
    # Prepare data for RAGAS evaluation
    ragas_data = pd.DataFrame({
        "user_input": results_df['user_input'],
        "reference": results_df['reference'],
        "response": results_df['response'].apply(lambda x: "" if x is None else x),
        "retrieved_contexts": results_df['context'].apply(lambda x: x if isinstance(x, list) else [])
    })
    
    # Create EvaluationDataset
    eval_dataset = EvaluationDataset.from_pandas(ragas_data)
    
    # Define metrics (non-LLM metrics first)
    metrics = [
        LenientFactualCorrectness(),
        SemanticSimilarity(embeddings=evaluator_embeddings),
        BleuScore(),
        RougeScore()
    ]
    
    # Add LLM-dependent metrics only if LLM is available
    if evaluator_llm is not None:
        metrics.extend([
            LLMContextRecall(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm)
        ])
    
    print(ragas_data[['user_input', 'response', 'reference']])
    
    # Run evaluation
    ragas_results = evaluate(
        eval_dataset,
        metrics=metrics
    )
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/synthetic_ragas_" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Save RAGAS results
    ragas_df = ragas_results.to_pandas()
    ragas_df.to_csv(output_dir / "ragas_results.csv", index=False)
    
    # Print results summary
    print("\nRAGAS Results:")
    for metric, value in ragas_df.mean().items():
        if metric != "hash":
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    run_synthetic_evaluation() 