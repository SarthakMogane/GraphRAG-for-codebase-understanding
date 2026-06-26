-- migrate:up
CREATE TABLE repo_scout_results (
    id                  UUID        PRIMARY KEY DEFAULT uuidv7(),
    repo_id             UUID         NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    account_id          UUID        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    
    head_sha            TEXT        NOT NULL,
    scout_json          JSONB       NOT NULL,                   -- full RepoScoutResult
    api_calls_made      INT,
    duration_ms         INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Only one scout result per commit per repo
    UNIQUE(repo_id, head_sha)
);

-- Enable RLS & Add Policy
ALTER TABLE repo_scout_results ENABLE ROW LEVEL SECURITY , FORCE ROW LEVEL SECURITY;
CREATE POLICY scout_results_isolation_policy ON repo_scout_results
    FOR ALL TO fastapi_app_user
    USING (
        account_id = current_setting('app.current_account_id', true)::uuid
        OR current_setting('app.is_system_flow', true) = 'true'
    )
    WITH CHECK (
        account_id = current_setting('app.current_account_id', true)::uuid
        OR current_setting('app.is_system_flow', true) = 'true'
    );

-- migrate:down
ALTER TABLE repo_scout_results DISABLE ROW LEVEL SECURITY , NO FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS scout_results_isolation_policy;
DROP TABLE IF EXISTS repo_scout_results;
