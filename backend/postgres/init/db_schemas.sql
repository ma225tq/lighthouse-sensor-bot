--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2 (Debian 17.2-1.pgdg120+1)
-- Dumped by pg_dump version 17.2

-- Started on 2025-04-15 13:51:58 UTC

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 245 (class 1255 OID 50151)
-- Name: refresh_full_query_data(); Type: FUNCTION; Schema: public; Owner: postgres
--

-- ***** MODIFIED FUNCTION START *****
CREATE OR REPLACE FUNCTION public.refresh_full_query_data() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- Check if the core driving table has any data. If not, exit early.
    IF NOT EXISTS (SELECT 1 FROM public.query_result LIMIT 1) THEN
        RAISE NOTICE 'Trigger % on table %: Skipping refresh of full_query_data as query_result table appears empty.', TG_NAME, TG_TABLE_NAME;
        RETURN NULL; -- Exit without attempting refresh
    END IF;

    -- If data exists, proceed with the refresh
    RAISE NOTICE 'Trigger % on table %: Refreshing materialized view full_query_data.', TG_NAME, TG_TABLE_NAME;
    REFRESH MATERIALIZED VIEW public.full_query_data;
    RETURN NULL; -- Important for AFTER triggers
END;
$$;
-- ***** MODIFIED FUNCTION END *****


ALTER FUNCTION public.refresh_full_query_data() OWNER TO postgres;

--
-- TOC entry 246 (class 1255 OID 50925)
-- Name: refresh_model_performance_metrics(); Type: FUNCTION; Schema: public; Owner: postgres
--

-- ***** MODIFIED FUNCTION START *****
CREATE OR REPLACE FUNCTION public.refresh_model_performance_metrics() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- Check if the core driving table has any data. If not, exit early.
    -- We check query_result as model performance metrics depend on evaluated queries.
    IF NOT EXISTS (SELECT 1 FROM public.query_result LIMIT 1) THEN
        RAISE NOTICE 'Trigger % on table %: Skipping refresh of model_performance_metrics as query_result table appears empty.', TG_NAME, TG_TABLE_NAME;
        RETURN NULL; -- Exit without attempting refresh
    END IF;

    -- If data exists, proceed with the refresh attempts
    RAISE NOTICE 'Trigger % on table %: Attempting concurrent refresh of materialized view model_performance_metrics.', TG_NAME, TG_TABLE_NAME;
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY public.model_performance_metrics;
        RAISE NOTICE 'Trigger % on table %: Concurrent refresh successful.', TG_NAME, TG_TABLE_NAME;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'Trigger % on table %: Could not refresh materialized view model_performance_metrics concurrently: %', TG_NAME, TG_TABLE_NAME, SQLERRM;
        BEGIN
            RAISE NOTICE 'Trigger % on table %: Attempting non-concurrent refresh of materialized view model_performance_metrics.', TG_NAME, TG_TABLE_NAME;
            REFRESH MATERIALIZED VIEW public.model_performance_metrics;
             RAISE NOTICE 'Trigger % on table %: Non-concurrent refresh successful.', TG_NAME, TG_TABLE_NAME;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Trigger % on table %: Could not refresh materialized view model_performance_metrics (non-concurrently): %', TG_NAME, TG_TABLE_NAME, SQLERRM;
        END;
    END;

    RETURN NULL; -- Important for AFTER triggers
END;
$$;
-- ***** MODIFIED FUNCTION END *****


ALTER FUNCTION public.refresh_model_performance_metrics() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 240 (class 1259 OID 24620)
-- Name: evaluation_metrics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.evaluation_metrics (
    id integer NOT NULL,
    factual_correctness numeric,
    semantic_similarity numeric,
    context_recall numeric,
    faithfulness numeric,
    bleu_score numeric,
    non_llm_string_similarity numeric,
    rogue_score numeric,
    string_present numeric
);


ALTER TABLE public.evaluation_metrics OWNER TO postgres;

--
-- TOC entry 239 (class 1259 OID 24619)
-- Name: evaluation_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.evaluation_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.evaluation_metrics_id_seq OWNER TO postgres;

--
-- TOC entry 3442 (class 0 OID 0)
-- Dependencies: 239
-- Name: evaluation_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.evaluation_metrics_id_seq OWNED BY public.evaluation_metrics.id;


--
-- TOC entry 234 (class 1259 OID 24578)
-- Name: llm_models; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.llm_models (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    type character varying NOT NULL,
    CONSTRAINT type_check CHECK (((type)::text = ANY ((ARRAY['proprietary'::character varying, 'open source'::character varying])::text[])))
);


ALTER TABLE public.llm_models OWNER TO postgres;

--
-- TOC entry 238 (class 1259 OID 24601)
-- Name: query_evaluation; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.query_evaluation (
    id integer NOT NULL,
    query_result_id integer,
    evaluation_metrics_id integer,
    retrieved_contexts text,
    reference text
);


ALTER TABLE public.query_evaluation OWNER TO postgres;

--
-- TOC entry 236 (class 1259 OID 24592)
-- Name: query_result; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.query_result (
    id integer NOT NULL,
    llm_model_id integer NOT NULL,
    query text,
    direct_response text,
    full_response text,
    sql_queries text,
    "timestamp" timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.query_result OWNER TO postgres;

--
-- TOC entry 241 (class 1259 OID 49976)
-- Name: token_usage; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.token_usage (
    prompt_tokens numeric,
    completion_tokens numeric,
    total_tokens numeric,
    query_result_id integer,
    id integer NOT NULL
);


ALTER TABLE public.token_usage OWNER TO postgres;

--
-- TOC entry 243 (class 1259 OID 50111)
-- Name: full_query_data; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres
--

CREATE MATERIALIZED VIEW public.full_query_data AS
 SELECT qr.id AS query_result_id,
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
    em.rogue_score,
    em.string_present,
    m.name AS model_name,
    tu.id AS token_usage_id,
    tu.prompt_tokens,
    tu.completion_tokens,
    tu.total_tokens
   FROM ((((public.query_result qr
     JOIN public.query_evaluation qe ON ((qr.id = qe.query_result_id)))
     JOIN public.evaluation_metrics em ON ((qe.evaluation_metrics_id = em.id)))
     JOIN public.llm_models m ON ((qr.llm_model_id = m.id)))
     LEFT JOIN public.token_usage tu ON ((qr.id = tu.query_result_id)))
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.full_query_data OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 24577)
-- Name: llm_models_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.llm_models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.llm_models_id_seq OWNER TO postgres;

--
-- TOC entry 3443 (class 0 OID 0)
-- Dependencies: 233
-- Name: llm_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.llm_models_id_seq OWNED BY public.llm_models.id;


--
-- TOC entry 244 (class 1259 OID 50912)
-- Name: model_performance_metrics; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres
--

CREATE MATERIALIZED VIEW public.model_performance_metrics AS
 SELECT lm.id AS model_id,
    lm.name AS model_name,
    lm.type AS model_type,
    avg(em.factual_correctness) FILTER (WHERE (em.factual_correctness IS NOT NULL)) AS avg_factual_correctness,
    avg(em.semantic_similarity) FILTER (WHERE (em.semantic_similarity IS NOT NULL)) AS avg_semantic_similarity,
    avg(em.context_recall) FILTER (WHERE (em.context_recall IS NOT NULL)) AS avg_context_recall,
    avg(em.faithfulness) FILTER (WHERE (em.faithfulness IS NOT NULL)) AS avg_faithfulness,
    avg(em.bleu_score) FILTER (WHERE (em.bleu_score IS NOT NULL)) AS avg_bleu_score,
    avg(em.non_llm_string_similarity) FILTER (WHERE (em.non_llm_string_similarity IS NOT NULL)) AS avg_non_llm_string_similarity,
    avg(em.rogue_score) FILTER (WHERE (em.rogue_score IS NOT NULL)) AS avg_rogue_score,
    avg(em.string_present) FILTER (WHERE (em.string_present IS NOT NULL)) AS avg_string_present,
    stddev_samp(em.factual_correctness) FILTER (WHERE (em.factual_correctness IS NOT NULL)) AS stddev_factual_correctness,
    stddev_samp(em.semantic_similarity) FILTER (WHERE (em.semantic_similarity IS NOT NULL)) AS stddev_semantic_similarity,
    stddev_samp(em.context_recall) FILTER (WHERE (em.context_recall IS NOT NULL)) AS stddev_context_recall,
    stddev_samp(em.faithfulness) FILTER (WHERE (em.faithfulness IS NOT NULL)) AS stddev_faithfulness,
    stddev_samp(em.bleu_score) FILTER (WHERE (em.bleu_score IS NOT NULL)) AS stddev_bleu_score,
    stddev_samp(em.non_llm_string_similarity) FILTER (WHERE (em.non_llm_string_similarity IS NOT NULL)) AS stddev_non_llm_string_similarity,
    stddev_samp(em.rogue_score) FILTER (WHERE (em.rogue_score IS NOT NULL)) AS stddev_rogue_score,
    stddev_samp(em.string_present) FILTER (WHERE (em.string_present IS NOT NULL)) AS stddev_string_present,
    count(qe.id) AS query_evaluation_count,
    avg(tu.total_tokens) FILTER (WHERE (tu.total_tokens IS NOT NULL)) AS avg_total_tokens,
    avg(tu.prompt_tokens) FILTER (WHERE (tu.prompt_tokens IS NOT NULL)) AS avg_prompt_tokens,
    avg(tu.completion_tokens) FILTER (WHERE (tu.completion_tokens IS NOT NULL)) AS avg_completion_tokens
   FROM ((((public.llm_models lm
     JOIN public.query_result qr ON ((lm.id = qr.llm_model_id)))
     JOIN public.query_evaluation qe ON ((qr.id = qe.query_result_id)))
     LEFT JOIN public.evaluation_metrics em ON ((qe.evaluation_metrics_id = em.id)))
     LEFT JOIN public.token_usage tu ON ((qr.id = tu.query_result_id)))
  GROUP BY lm.id, lm.name, lm.type
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.model_performance_metrics OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 24600)
-- Name: query_evaluation_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.query_evaluation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.query_evaluation_id_seq OWNER TO postgres;

--
-- TOC entry 3444 (class 0 OID 0)
-- Dependencies: 237
-- Name: query_evaluation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.query_evaluation_id_seq OWNED BY public.query_evaluation.id;


--
-- TOC entry 235 (class 1259 OID 24591)
-- Name: query_result_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.query_result_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.query_result_id_seq OWNER TO postgres;

--
-- TOC entry 3445 (class 0 OID 0)
-- Dependencies: 235
-- Name: query_result_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.query_result_id_seq OWNED BY public.query_result.id;


--
-- TOC entry 242 (class 1259 OID 50024)
-- Name: token_usage_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.token_usage_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.token_usage_id_seq OWNER TO postgres;

--
-- TOC entry 3446 (class 0 OID 0)
-- Dependencies: 242
-- Name: token_usage_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.token_usage_id_seq OWNED BY public.token_usage.id;


--
-- TOC entry 3260 (class 2604 OID 24623)
-- Name: evaluation_metrics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evaluation_metrics ALTER COLUMN id SET DEFAULT nextval('public.evaluation_metrics_id_seq'::regclass);


--
-- TOC entry 3256 (class 2604 OID 24581)
-- Name: llm_models id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.llm_models ALTER COLUMN id SET DEFAULT nextval('public.llm_models_id_seq'::regclass);


--
-- TOC entry 3259 (class 2604 OID 24604)
-- Name: query_evaluation id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_evaluation ALTER COLUMN id SET DEFAULT nextval('public.query_evaluation_id_seq'::regclass);


--
-- TOC entry 3257 (class 2604 OID 24595)
-- Name: query_result id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_result ALTER COLUMN id SET DEFAULT nextval('public.query_result_id_seq'::regclass);


--
-- TOC entry 3261 (class 2604 OID 50025)
-- Name: token_usage id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.token_usage ALTER COLUMN id SET DEFAULT nextval('public.token_usage_id_seq'::regclass);


--
-- TOC entry 3272 (class 2606 OID 24627)
-- Name: evaluation_metrics evaluation_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evaluation_metrics
    ADD CONSTRAINT evaluation_metrics_pkey PRIMARY KEY (id);


--
-- TOC entry 3264 (class 2606 OID 24583)
-- Name: llm_models llm_models_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.llm_models
    ADD CONSTRAINT llm_models_pkey PRIMARY KEY (id);


--
-- TOC entry 3270 (class 2606 OID 24608)
-- Name: query_evaluation query_evaluation_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_evaluation
    ADD CONSTRAINT query_evaluation_pkey PRIMARY KEY (id);


--
-- TOC entry 3268 (class 2606 OID 24599)
-- Name: query_result query_result_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_result
    ADD CONSTRAINT query_result_pkey PRIMARY KEY (id);


--
-- TOC entry 3274 (class 2606 OID 50032)
-- Name: token_usage token_usage_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.token_usage
    ADD CONSTRAINT token_usage_pkey PRIMARY KEY (id);


--
-- TOC entry 3266 (class 2606 OID 24590)
-- Name: llm_models unique_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.llm_models
    ADD CONSTRAINT unique_name UNIQUE (name);


--
-- TOC entry 3275 (class 1259 OID 50157)
-- Name: idx_full_query_data_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_full_query_data_timestamp ON public.full_query_data USING btree (query_timestamp);


--
-- TOC entry 3276 (class 1259 OID 50924)
-- Name: model_performance_metrics_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX model_performance_metrics_idx ON public.model_performance_metrics USING btree (model_id);


--
-- TOC entry 3286 (class 2620 OID 50154)
-- Name: evaluation_metrics refresh_full_query_data_trigger_em; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_full_query_data_trigger_em AFTER INSERT OR DELETE OR UPDATE ON public.evaluation_metrics FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_full_query_data();


--
-- TOC entry 3281 (class 2620 OID 50155)
-- Name: llm_models refresh_full_query_data_trigger_lm; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_full_query_data_trigger_lm AFTER INSERT OR DELETE OR UPDATE ON public.llm_models FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_full_query_data();


--
-- TOC entry 3284 (class 2620 OID 50152)
-- Name: query_evaluation refresh_full_query_data_trigger_qe; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_full_query_data_trigger_qe AFTER INSERT OR DELETE OR UPDATE ON public.query_evaluation FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_full_query_data();


--
-- TOC entry 3283 (class 2620 OID 50153)
-- Name: query_result refresh_full_query_data_trigger_qr; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_full_query_data_trigger_qr AFTER INSERT OR DELETE OR UPDATE ON public.query_result FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_full_query_data();


--
-- TOC entry 3288 (class 2620 OID 50156)
-- Name: token_usage refresh_full_query_data_trigger_tu; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_full_query_data_trigger_tu AFTER INSERT OR DELETE OR UPDATE ON public.token_usage FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_full_query_data();


--
-- TOC entry 3287 (class 2620 OID 50928)
-- Name: evaluation_metrics refresh_model_performance_metrics_on_evaluation_metrics; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_model_performance_metrics_on_evaluation_metrics AFTER INSERT OR DELETE OR UPDATE ON public.evaluation_metrics FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_model_performance_metrics();


--
-- TOC entry 3282 (class 2620 OID 50927)
-- Name: llm_models refresh_model_performance_metrics_on_llm_models; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_model_performance_metrics_on_llm_models AFTER INSERT OR DELETE OR UPDATE ON public.llm_models FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_model_performance_metrics();


--
-- TOC entry 3285 (class 2620 OID 50926)
-- Name: query_evaluation refresh_model_performance_metrics_on_query_evaluation; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_model_performance_metrics_on_query_evaluation AFTER INSERT OR DELETE OR UPDATE ON public.query_evaluation FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_model_performance_metrics();


--
-- TOC entry 3289 (class 2620 OID 50929)
-- Name: token_usage refresh_model_performance_metrics_on_token_usage; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER refresh_model_performance_metrics_on_token_usage AFTER INSERT OR DELETE OR UPDATE ON public.token_usage FOR EACH STATEMENT EXECUTE FUNCTION public.refresh_model_performance_metrics();


--
-- TOC entry 3278 (class 2606 OID 24628)
-- Name: query_evaluation fk_query_evaluation_evaluation_metrics_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_evaluation
    ADD CONSTRAINT fk_query_evaluation_evaluation_metrics_id FOREIGN KEY (evaluation_metrics_id) REFERENCES public.evaluation_metrics(id) NOT VALID;


--
-- TOC entry 3279 (class 2606 OID 24609)
-- Name: query_evaluation fk_query_evaluation_query_result_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_evaluation
    ADD CONSTRAINT fk_query_evaluation_query_result_id FOREIGN KEY (query_result_id) REFERENCES public.query_result(id) NOT VALID;


--
-- TOC entry 3277 (class 2606 OID 24614)
-- Name: query_result fk_query_result_llm_model_id_llm_models_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_result
    ADD CONSTRAINT fk_query_result_llm_model_id_llm_models_id FOREIGN KEY (llm_model_id) REFERENCES public.llm_models(id) NOT VALID;


--
-- TOC entry 3280 (class 2606 OID 49985)
-- Name: token_usage token_usage_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.token_usage
    ADD CONSTRAINT token_usage_fk FOREIGN KEY (query_result_id) REFERENCES public.query_result(id) NOT VALID;


-- Completed on 2025-04-15 13:51:58 UTC

--
-- PostgreSQL database dump complete
--