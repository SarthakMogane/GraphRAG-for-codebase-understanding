-- migrate:up

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE installations (
    id                  UUID        PRIMARY KEY DEFAULT uuidv7(),
    account_id          UUID        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,

    -- GitHub App identity
    github_install_id   BIGINT      NOT NULL UNIQUE,            -- GitHub's installation ID
    owner_login         CITEXT      NOT NULL,                   -- org or user name
    owner_type          TEXT        NOT NULL,                   -- 'User' | 'Organization'
    owner_github_id     BIGINT      NOT NULL,

    -- Cached installation token (refreshed by GitHubAuthManager)
    install_token_enc   BYTEA,                                  -- encrypted
    token_expires_at    TIMESTAMPTZ,

    -- Lifecycle
    is_active           BOOL        NOT NULL DEFAULT TRUE,
    installed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    suspended_at        TIMESTAMPTZ,
    uninstalled_at      TIMESTAMPTZ
);

---RLS 
ALTER installations ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY;

--POLICY

CREATE POLICY installation_isolation ON installations
    FOR ALL TO fastapi_app_user
    USING (account_id = current_settings('app.current_account_id',true)::uuid
             -- Condition 2: Automated Background Worker (SQS/Celery traffic)
            OR current_setting('app.is_system_flow', true) = 'true'
    );


-- migrate:down

ALTER installation DISABLE ROW LEVEL SECURITY , NO FORCE ROW LEVEL SECURITY  ;

DROP POLICY IF EXISTS installation_isolation;
DROP TABLE IF EXISTS installations;
DROP EXTENSION IF NOT EXISTS citext;

