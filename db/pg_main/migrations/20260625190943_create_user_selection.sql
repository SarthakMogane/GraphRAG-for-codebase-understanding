-- migrate:up
CREATE TABLE user_selections (
    id                      UUID        PRIMARY KEY DEFAULT uuidv7(),
    repo_id                 UUID         NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    account_id              UUID        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    created_by_user_id      UUID        NOT NULL REFERENCES users(id), -- Tracks WHO made the edit
    scout_result_id         UUID        REFERENCES repo_scout_results(id) ON DELETE SET NULL,

    selected_subprojects    TEXT[]      NOT NULL DEFAULT '{}',
    selected_submodules     TEXT[]      NOT NULL DEFAULT '{}',
    deselected_subprojects  TEXT[]      NOT NULL DEFAULT '{}',
    deselected_submodules   TEXT[]      NOT NULL DEFAULT '{}',

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Crucial: Only ONE active configuration per repository.
    -- When a user updates their selection, we UPDATE this row, we don't insert a new one.
    UNIQUE(repo_id) 
);

-- Enable RLS & Add Policy
ALTER TABLE user_selections ENABLE ROW LEVEL SECURITY , FORCE ROW LEVEL SECURITY;

CREATE POLICY selections_isolation_policy ON user_selections
    FOR ALL TO fastapi_app_user
    USING (
        account_id = current_setting('app.current_account_id',true)::uuid
        OR 
        current_setting('app.is_system_flow',true) = 'true'
    )
    WITH CHECK (
        account_id = current_setting('app.current_id',true)::uuid
        OR 
        current_setting('app.is_system_flow',true) = 'true'
    );



-- migrate:down
ALTER TABLE user_selections DISABLE ROW LEVEL SECURITY , NO FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS selections_isolation_policy;
DROP TABLE IF EXISTS user_selections;

