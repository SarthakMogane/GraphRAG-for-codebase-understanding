-- migrate:up

CREATE EXTENSION IF NOT EXISTS citext;
---create table 
CREATE TABLE repos (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id          UUID        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    installation_id     UUID        NOT NULL REFERENCES installations(id) ON DELETE CASCADE,

    -- GitHub identity
    github_repo_id      BIGINT      NOT NULL UNIQUE,
    full_name           CITEXT      NOT NULL,                   -- "owner/repo"
    owner_login         CITEXT      NOT NULL,
    repo_name           TEXT        NOT NULL,
    private             BOOL        NOT NULL DEFAULT TRUE,
    is_fork             BOOL        NOT NULL DEFAULT FALSE,
    upstream_repo_id    UUID        REFERENCES repos(id),       -- for forks
    default_branch      TEXT        NOT NULL DEFAULT 'main',
    primary_language    TEXT,
    size_kb             INT,

    -- Structure detected during scout
    structure_type      structure_type NOT NULL DEFAULT 'flat',
    is_monorepo         BOOL        NOT NULL DEFAULT FALSE,
    monorepo_tooling    TEXT,                                   -- 'nx' | 'turborepo' | 'pnpm' etc.

    -- Indexing state
    index_status        repo_status NOT NULL DEFAULT 'not_indexed',
    index_sha           TEXT,                                   -- HEAD SHA at last successful index
    last_indexed_at     TIMESTAMPTZ,
    last_scout_sha      TEXT,                                   -- HEAD SHA at last scout
    last_scout_at       TIMESTAMPTZ,
    last_stale_type     staleness_type,                        -- last stale check result

    -- Clone strategy used (from ingestion)
    clone_strategy      TEXT,                                   -- 'shallow' | 'partial_blob' | 'sparse'

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- State Machine Constraint
    CONSTRAINT valid_index_status CHECK (
        index_status IN (
            'not_indexed',   -- Brand new, waiting for user to click Index
            'indexing',      -- Worker is currently running Tree-sitter
            'ready',         -- GraphRAG is fully populated and queryable
            'stale',         -- Code changed, needs re-indexing
            'inaccessible'   -- User uninstalled app or removed repo access
        )
    )
);


--- create policy 

ALTER repos ENABLE ROW LEVEL SECURITY,FORCE ROW LEVEL SECURITY

CREATE POLICY repos_tenant_isolation ON repos
FOR ALL TO fastapi_app_user
USING(
    account_id = current_setting("app.current_account_id",true)::uuid
    OR 
    current_setting('app.is_system_flow',true)="true"
)

-- migrate:down

ALTER repos DISABLE ROW LEVEL SECURITY,NO FORCE ROW LEVEL SECURITY
DROP POLICY IF EXISTS repos_tenant_isolation
DROP TABLE IF EXISTS repos
DROP EXTENSION IF EXISTS citext



