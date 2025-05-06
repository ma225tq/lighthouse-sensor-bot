from typing import Dict, Optional, List, Union, Tuple, Any
from agno.utils.log import logger
from app.conf.postgres import get_cursor
import ast




def get_model_id(llm_model_name: str) -> int:
    with get_cursor() as cursor:
        cursor.execute("SELECT id FROM llm_models WHERE name = %s", (llm_model_name,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No model found with name: {llm_model_name}")
        return result[0]


def save_query_to_db(
    query: str,
    direct_response: str,
    full_response: str,
    llm_model_id: str,
    sql_queries: Optional[List[str]] = None,
    token_usage: Optional[Dict[str, int]] = None,
) -> int:
    """Save the query and response to the database."""
    with get_cursor() as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO query_result (
                    query, direct_response, full_response, sql_queries, llm_model_id
                ) VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (
                    query,
                    direct_response,
                    full_response,
                    sql_queries,
                    get_model_id(llm_model_id),
                ),
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError("Failed to save query: No ID returned")
            query_result_id = result[0]
            
            # Save token usage in the separate table if provided
            if token_usage:
                cursor.execute(
                    """
                    INSERT INTO token_usage (
                        query_result_id, prompt_tokens, completion_tokens, total_tokens
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        query_result_id,
                        token_usage.get('prompt_tokens'),
                        token_usage.get('completion_tokens'),
                        token_usage.get('total_tokens'),
                    ),
                )
            
            return query_result_id
        except Exception as e:
            logger.error(f"Error saving query to database: {e}")
            raise
          



def save_query_with_eval_to_db(
    query: str,
    direct_response: str,
    full_response: str,
    llm_model_id: str,
    evaluation_results: Dict[str, Union[str, float, int, bool, None]],
    sql_queries: Optional[List[str]] = None,
    token_usage: Optional[Dict[str, int]] = None,
) -> None:
    """Save the query and response to the database with evaluation results."""
    if not isinstance(evaluation_results, dict):
      raise ValueError("Evaluation results must be a dictionary")

    required_keys = {
        "retrieved_contexts": (str, type(None)),
        "reference": (str, type(None)),
        "factual_correctness": (float, int, type(None)),
        "semantic_similarity": (float, int, type(None)),
        "context_recall": (float, int, type(None)),
        "faithfulness": (float, int, type(None)),
        "bleu_score": (float, int, type(None)),
        "non_llm_string_similarity": (float, int, type(None)),
        "rouge_score": (float, int, type(None)),
        "string_present": (float, int, type(None)),
    }

    for key, expected_types in required_keys.items():
        value = evaluation_results.get(key)
        if value is not None and not isinstance(value, expected_types):
            raise ValueError(
                f"Invalid type for {key}: expected {expected_types}, got {type(value)}"
            )

    # First save the query to get query_result_id
    query_result_id = save_query_to_db(
        query, direct_response, full_response, llm_model_id, sql_queries, token_usage
    )

    with get_cursor() as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO evaluation_metrics (
                    factual_correctness, semantic_similarity, context_recall, 
                    faithfulness, bleu_score, non_llm_string_similarity,
                    rouge_score, string_present
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (
                    evaluation_results.get("factual_correctness"),
                    evaluation_results.get("semantic_similarity"),
                    evaluation_results.get("context_recall"),
                    evaluation_results.get("faithfulness"),
                    evaluation_results.get("bleu_score"),
                    evaluation_results.get("non_llm_string_similarity"),
                    evaluation_results.get("rouge_score"),
                    evaluation_results.get("string_present"),
                ),
            )
            evaluation_metrics_id = cursor.fetchone()[0]
            
            # Then insert into query_evaluation with reference to evaluation_metrics
            cursor.execute(
                """
                INSERT INTO query_evaluation (
                    retrieved_contexts, reference, query_result_id, evaluation_metrics_id
                ) VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (
                    evaluation_results.get("retrieved_contexts"),
                    evaluation_results.get("reference"),
                    query_result_id,
                    evaluation_metrics_id,
                ),
            )
        except Exception as e:
            logger.error(f"Error saving evaluation results to database: {e}")
            raise