DROP TRIGGER IF EXISTS refresh_model_performance_metrics_on_query_evaluation ON public.query_evaluation;
DROP TRIGGER IF EXISTS refresh_model_performance_metrics_on_llm_models ON public.llm_models;
DROP TRIGGER IF EXISTS refresh_model_performance_metrics_on_evaluation_metrics ON public.evaluation_metrics;
DROP TRIGGER IF EXISTS refresh_model_performance_metrics_on_token_usage ON public.token_usage;

DROP FUNCTION IF EXISTS refresh_model_performance_metrics();

DROP MATERIALIZED VIEW IF EXISTS model_performance_metrics;

CREATE MATERIALIZED VIEW model_performance_metrics AS
SELECT
    lm.id as model_id,
    lm.name as model_name,
    lm.type as model_type,

    AVG(em.factual_correctness) FILTER (WHERE em.factual_correctness IS NOT NULL) as avg_factual_correctness,
    AVG(em.semantic_similarity) FILTER (WHERE em.semantic_similarity IS NOT NULL) as avg_semantic_similarity,
    AVG(em.context_recall) FILTER (WHERE em.context_recall IS NOT NULL) as avg_context_recall,
    AVG(em.faithfulness) FILTER (WHERE em.faithfulness IS NOT NULL) as avg_faithfulness,
    AVG(em.bleu_score) FILTER (WHERE em.bleu_score IS NOT NULL) as avg_bleu_score,
    AVG(em.non_llm_string_similarity) FILTER (WHERE em.non_llm_string_similarity IS NOT NULL) as avg_non_llm_string_similarity,
    AVG(em.rouge_score) FILTER (WHERE em.rouge_score IS NOT NULL) as avg_rouge_score,
    AVG(em.string_present) FILTER (WHERE em.string_present IS NOT NULL) as avg_string_present, 

    STDDEV_SAMP(em.factual_correctness) FILTER (WHERE em.factual_correctness IS NOT NULL) as stddev_factual_correctness,
    STDDEV_SAMP(em.semantic_similarity) FILTER (WHERE em.semantic_similarity IS NOT NULL) as stddev_semantic_similarity,
    STDDEV_SAMP(em.context_recall) FILTER (WHERE em.context_recall IS NOT NULL) as stddev_context_recall,
    STDDEV_SAMP(em.faithfulness) FILTER (WHERE em.faithfulness IS NOT NULL) as stddev_faithfulness,
    STDDEV_SAMP(em.bleu_score) FILTER (WHERE em.bleu_score IS NOT NULL) as stddev_bleu_score,
    STDDEV_SAMP(em.non_llm_string_similarity) FILTER (WHERE em.non_llm_string_similarity IS NOT NULL) as stddev_non_llm_string_similarity,
    STDDEV_SAMP(em.rouge_score) FILTER (WHERE em.rouge_score IS NOT NULL) as stddev_rouge_score,
    STDDEV_SAMP(em.string_present) FILTER (WHERE em.string_present IS NOT NULL) as stddev_string_present,

    COUNT(qe.id) as query_evaluation_count,
    AVG(tu.total_tokens) FILTER (WHERE tu.total_tokens IS NOT NULL) as avg_total_tokens,
    AVG(tu.prompt_tokens) FILTER (WHERE tu.prompt_tokens IS NOT NULL) as avg_prompt_tokens,
    AVG(tu.completion_tokens) FILTER (WHERE tu.completion_tokens IS NOT NULL) as avg_completion_tokens
FROM
    llm_models lm
INNER JOIN
    public.query_result qr ON lm.id = qr.llm_model_id
INNER JOIN
    public.query_evaluation qe ON qr.id = qe.query_result_id
LEFT JOIN -- Using LEFT JOIN for metrics is good in case metrics are sometimes missing for an evaluation
    public.evaluation_metrics em ON qe.evaluation_metrics_id = em.id
LEFT JOIN
    public.token_usage tu ON qr.id = tu.query_result_id
GROUP BY
    lm.id, lm.name, lm.type
WITH DATA;

DROP INDEX IF EXISTS model_performance_metrics_idx;
CREATE UNIQUE INDEX model_performance_metrics_idx ON model_performance_metrics (model_id);

CREATE OR REPLACE FUNCTION refresh_model_performance_metrics()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY model_performance_metrics;
    RETURN NULL;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Could not refresh materialized view model_performance_metrics concurrently: %', SQLERRM;
    BEGIN
        REFRESH MATERIALIZED VIEW model_performance_metrics;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'Could not refresh materialized view model_performance_metrics: %', SQLERRM;
    END;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_model_performance_metrics_on_query_evaluation
AFTER INSERT OR UPDATE OR DELETE ON query_evaluation
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_model_performance_metrics();

CREATE TRIGGER refresh_model_performance_metrics_on_llm_models
AFTER INSERT OR UPDATE OR DELETE ON llm_models
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_model_performance_metrics();

CREATE TRIGGER refresh_model_performance_metrics_on_evaluation_metrics
AFTER INSERT OR UPDATE OR DELETE ON evaluation_metrics
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_model_performance_metrics();

CREATE TRIGGER refresh_model_performance_metrics_on_token_usage
AFTER INSERT OR UPDATE OR DELETE ON public.token_usage
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_model_performance_metrics();