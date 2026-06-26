-- migrate:up
-- INGESTION JOBS (Outbox & SQS Optimized)
-- -----------------------------------------------------------------------------
CREATE TABLE ingestion_jobs (
    id                  UUID        PRIMARY KEY DEFAULT uuidv7(),
    repo_id             UUID         NOT NULL REFERENCES repos(id) ON DELETE CASCADE, 
    account_id          UUID        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    selection_id        UUID        REFERENCES user_selections(id) ON DELETE SET NULL,

    job_type            job_type    NOT NULL,
    status              job_status  NOT NULL DEFAULT 'dispatch_pending', 

    -- Staleness context (for REFRESH jobs)
    trigger_sha_before  TEXT,
    trigger_sha_after   TEXT,
    changed_files       TEXT[],
    stale_type          staleness_type,

    -- Results
    files_total         INT,
    files_accepted      INT,
    files_excluded      INT,
    submodules_found    INT,
    submodules_indexed  INT,

    -- Timing (Crucial for Outbox Relay & Janitor)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,

    -- Cost tracking (BYOK & Billing)
    embedding_tokens    BIGINT      NOT NULL DEFAULT 0,
    llm_tokens_in       BIGINT      NOT NULL DEFAULT 0,
    llm_tokens_out      BIGINT      NOT NULL DEFAULT 0
);

ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY , FORCE ROW LEVEL SECURITY;

-- Create policy 
CREATE POLICY ingestion_jobs_isolation_policy ON ingestion_jobs
FOR ALL 
TO fastapi_app_user
USING (
    account_id = current_settings("app.current_account_id",TRUE)::uuid
    OR
    current_setting("app.is_system_flow",TRUE) = "true")
WITH CHECK (
    account_id = current_setting("app.current_account_id",TRUE)::uuid
    OR 
    current_setting("app.is_system_flow",TRUE)= "true");

-- migrate:down

ALTER ingestion_jobs DISABLE ROW LEVEL SECURITY , NO FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS ingestion_jobs_isolation_policy;
DROP TABLE IF EXISTS ingestion_jobs;