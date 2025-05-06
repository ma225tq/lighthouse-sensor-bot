DROP TRIGGER IF EXISTS refresh_full_query_data_trigger_qe ON public.query_evaluation;
DROP TRIGGER IF EXISTS refresh_full_query_data_trigger_qr ON public.query_result;
DROP TRIGGER IF EXISTS refresh_full_query_data_trigger_em ON public.evaluation_metrics;
DROP TRIGGER IF EXISTS refresh_full_query_data_trigger_lm ON public.llm_models;
DROP TRIGGER IF EXISTS refresh_full_query_data_trigger_tu ON public.token_usage;

DROP FUNCTION IF EXISTS refresh_full_query_data();

DROP MATERIALIZED VIEW IF EXISTS full_query_data;

CREATE MATERIALIZED VIEW full_query_data AS
SELECT
    qr.id AS query_result_id,
    qr.query,
    qr.direct_response,
    qr.full_response,
    qr.sql_queries,
    qr."timestamp" AS query_timestamp,
    em.id AS evaluation_metric_id,
    em.factual_correctness,
    em.semantic_similarity,
    em.context_recall,
    em.faithfulness,
    em.bleu_score,
    em.non_llm_string_similarity,
    em.rouge_score,
    em.string_present,
    m.name AS model_name,
    tu.id AS token_usage_id,
    tu.prompt_tokens,
    tu.completion_tokens,
    tu.total_tokens
FROM
    public.query_result qr
JOIN
    public.query_evaluation qe ON qr.id = qe.query_result_id
JOIN
    public.evaluation_metrics em ON qe.evaluation_metrics_id = em.id
JOIN
    public.llm_models m ON qr.llm_model_id = m.id
LEFT JOIN -- Use LEFT JOIN in case some query results don't have token usage yet
    public.token_usage tu ON qr.id = tu.query_result_id
WITH DATA;

CREATE OR REPLACE FUNCTION refresh_full_query_data()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW full_query_data;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_full_query_data_trigger_qe
AFTER INSERT OR UPDATE OR DELETE ON public.query_evaluation
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_full_query_data();

CREATE TRIGGER refresh_full_query_data_trigger_qr
AFTER INSERT OR UPDATE OR DELETE ON public.query_result
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_full_query_data();

CREATE TRIGGER refresh_full_query_data_trigger_em
AFTER INSERT OR UPDATE OR DELETE ON public.evaluation_metrics
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_full_query_data();

CREATE TRIGGER refresh_full_query_data_trigger_lm
AFTER INSERT OR UPDATE OR DELETE ON public.llm_models
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_full_query_data();

CREATE TRIGGER refresh_full_query_data_trigger_tu
AFTER INSERT OR UPDATE OR DELETE ON public.token_usage
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_full_query_data();

DROP INDEX IF EXISTS idx_full_query_data_id;
CREATE INDEX idx_full_query_data_timestamp ON full_query_data (query_timestamp);