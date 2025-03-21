import pandas as pd
import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, SemanticSimilarity
# from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RougeScore
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_openai import ChatOpenAI
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

# Initialize LLM and Embeddings wrappers using the same models as in ragas_tests.py
evaluator_llm = LangchainLLMWrapper(
    HuggingFaceEndpoint(
        repo_id="google/flan-t5-xl",  # Same model as in ragas_tests.py
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # This is completely FREE - it runs through Hugging Face's free API.
)

def run_test_case(query, ground_truth=None):
    """Run a single test case through the API"""
    api_url = f"{API_URL}/api/query"
    try:
        response = requests.post(api_url, json={
            "question": query,
            "source_file": "ferry_trips_data.csv"
        })
        response.raise_for_status()
        
        response_data = response.json()
        agent_response = response_data.get('content')
        full_response = response_data.get('full_response')
        sql_queries = response_data.get('sql_queries', [])
        
        if agent_response is None:
            print(f"Error: No 'content' key found in the API response for query: {query}")
            return None, None, False
            
        # Format contexts including SQL queries and full response
        contexts = []
        for sql in sql_queries:
            contexts.append(f"SQL Query: {sql}")
        contexts.append(f"Agent Reasoning and Response: {full_response}")
        
        return agent_response, contexts, True
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for query: {query}: {e}")
        return None, None, False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for query: {query}: {e}")
        return None, None, False

def load_synthetic_test_cases():
    """Load synthetic test cases from JSON file"""
    test_cases_path = Path("app/ragas/test_cases/synthetic_test_cases.json")
    try:
        with open(test_cases_path, "r") as f:
            test_cases = json.load(f)
        return test_cases
    except FileNotFoundError:
        print(f"Error: {test_cases_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {test_cases_path}.")
        return None

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
        
        if api_call_success:
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
        "response": results_df['response'],
        "retrieved_contexts": results_df['reference_contexts']
    })
    
    # Create EvaluationDataset
    eval_dataset = EvaluationDataset.from_pandas(ragas_data)
    
    # Define metrics
    metrics = [
        LenientFactualCorrectness(),
        SemanticSimilarity(embeddings=evaluator_embeddings),
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        BleuScore(),
        RougeScore()
    ]
    
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
    
    # Save metrics summary
    metrics_df = ragas_results.to_pandas()
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    
    # Save combined results with metrics
    for metric_name, scores in ragas_results.to_pandas().items():
        if metric_name != 'hash':
            results_df[metric_name] = scores
    results_df.to_csv(output_dir / "combined_results.csv", index=False)
    
    print("\nEvaluation Results:")
    print(ragas_results)
    print(f"\nDetailed results saved to: {output_dir}")
    
    return ragas_results, results_df

if __name__ == "__main__":
    print("Starting synthetic dataset evaluation...")
    run_synthetic_evaluation() 