import pandas as pd
import json
import requests
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AspectCritic, LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import datetime
import argparse
from app.ragas.custom_metrics.LenientFactualCorrectness import LenientFactualCorrectness
from ragas.metrics._string import NonLLMStringSimilarity 
from ragas.metrics import RougeScore, StringPresence
from app.ragas.custom_metrics.bleu_score import BleuScore
import numpy as np
import math

load_dotenv()
# This is the main RAGAS Evaluation script. It is used to evaluate the performance of the agent.
API_URL = os.getenv('API_URL')
RAGAS_APP_TOKEN = os.getenv('RAGAS_APP_TOKEN')

# Initialize the DeepSeek LLM instead of OpenAI - this is free and works well
evaluator_llm = LangchainLLMWrapper(ChatDeepSeek(model="deepseek-chat", temperature=0))

# Use HuggingFace embeddings instead of OpenAI embeddings
evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Free, high-quality embedding model
))

def run_test_case(query, ground_truth, llm_model_id):
    api_url = f"{API_URL}/api/query"
    try:
        response = requests.post(api_url, json={
            "question": query,
            "source_file": "ferry_trips_data.csv",
            "llm_model_id": llm_model_id
        })
        response.raise_for_status()
        
        response_data = response.json()
        agent_response = response_data.get('content')  # This is already the clean response
        full_response = response_data.get('full_response')  # This contains the full context
        sql_queries = response_data.get('sql_queries', [])
        
        if agent_response is None:
            print(f"Error: No 'content' key found in the API response for query: {query}")
            return None, None, True
            
        # Format the complete context with reasoning and SQL
        contexts = []
        for sql in sql_queries:
            contexts.append(f"SQL Query: {sql}")
        
        # Add the complete agent response as context
        contexts.append(f"Agent Reasoning and Response: {full_response}")
        
        return agent_response, contexts, True
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for query: {query}: {e}")
        return None, None, False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for query: {query}: {e}")
        return None, None, False

def run_evaluation(llm_model_id):
    # Load test cases from JSON file
    test_cases_path = Path("app/ragas/test_cases/test_cases.json")
    try:
        with open(test_cases_path, "r") as f:
            test_cases_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: {test_cases_path} not found.")
        return {"error": f"Test cases file not found"}, pd.DataFrame()
    
    results = []
    for test_case in test_cases_list:
        query = test_case['user_input']
        ground_truth = test_case.get('reference')
        response, context, api_call_success = run_test_case(query, ground_truth, llm_model_id)
        results.append({
            "user_input": query,
            "reference": ground_truth,
            "response": response,
            "context": context,
            "api_call_success": api_call_success
        })

    results_df = pd.DataFrame(results)
    
    # RAGAS Evaluation with all metrics but using free models
    try:
        ragas_data = pd.DataFrame({
            "user_input": results_df['user_input'],
            "reference": results_df['reference'],
            "response": results_df['response'],
            "retrieved_contexts": results_df['context'].apply(lambda x: x if isinstance(x, list) else [])
        })
        
        # Create EvaluationDataset
        eval_dataset = EvaluationDataset.from_pandas(ragas_data)
        
        # Include ALL important metrics but using free models
        metrics = [
            # Core metrics you want
            LenientFactualCorrectness(),
            SemanticSimilarity(embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),  # Using DeepSeek instead of OpenAI
            LLMContextRecall(llm=evaluator_llm),  # Using DeepSeek instead of OpenAI
            
            # Additional string-based metrics that don't need LLMs
            BleuScore(),
            NonLLMStringSimilarity(),
            RougeScore(),
            StringPresence()
        ]
        
        # Run the evaluation using DeepSeek LLM
        ragas_results = evaluate(eval_dataset, metrics, llm=evaluator_llm)
        
        # We'll skip the upload since that requires API key
        # ragas_results.upload()
        
        # Add RAGAS metrics to results_df
        for metric_name, scores in ragas_results.to_pandas().items():
            if metric_name != 'hash':
                results_df[metric_name] = scores
                
        # Calculate average scores
        metric_scores = {}
        for column in results_df.columns:
            if column not in ['user_input', 'reference', 'response', 'context', 'api_call_success']:
                try:
                    # Get mean value
                    mean_val = results_df[column].mean()
                    
                    # Replace NaN or inf with 0
                    if math.isnan(mean_val) or math.isinf(mean_val):
                        metric_scores[column] = 0.0
                    else:
                        metric_scores[column] = float(mean_val)
                except:
                    metric_scores[column] = 0.0
        
        # Make sure string_present is a Boolean
        if "string_present" in metric_scores:
            try:
                # Convert to boolean properly
                val = metric_scores["string_present"]
                if isinstance(val, (np.bool_, bool)):
                    metric_scores["string_present"] = bool(val)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    metric_scores["string_present"] = bool(val > 0.5)
                else:
                    metric_scores["string_present"] = False
            except:
                metric_scores["string_present"] = False
        
        # Add default values for any missing keys to prevent DB errors
        expected_keys = [
            "factual_correctness", "semantic_similarity", "context_recall", 
            "faithfulness", "bleu_score", "non_llm_string_similarity", 
            "rogue_score", "string_present"
        ]
        
        for key in expected_keys:
            if key not in metric_scores:
                if key == "string_present":
                    metric_scores[key] = False
                else:
                    metric_scores[key] = 0.0
                    
        metric_scores["retrieved_contexts"] = None
        metric_scores["reference"] = None
        
        return metric_scores, results_df
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Return default values for all required metrics
        return {
            "error": str(e),
            "factual_correctness": 0.0,
            "semantic_similarity": 0.0, 
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "bleu_score": 0.0,
            "non_llm_string_similarity": 0.0,
            "rogue_score": 0.0,
            "string_present": False,
            "retrieved_contexts": None,
            "reference": None
        }, results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--model_id", type=str, help="Model ID for evaluation")
    
    args = parser.parse_args()
    
    if args.model_id:
        # Run evaluation directly if model_id is provided
        results, df = run_evaluation(args.model_id)
        print("Evaluation results:", results)
    else:
        # Wait for model_id from frontend through the API
        print("Evaluation server started. Waiting for frontend model selection...")
        print("To run evaluation directly, use: make eval MODEL_ID=<model_id>")
        
        # Keep the script running
        import time
        try:
            while True:
                time.sleep(10)  # Sleep to prevent high CPU usage
        except KeyboardInterrupt:
            print("Evaluation server stopped.")