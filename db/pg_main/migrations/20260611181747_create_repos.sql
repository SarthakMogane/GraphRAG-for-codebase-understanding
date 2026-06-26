-- migrate:up

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TYPE repo_status   AS ENUM (
    'not_indexed', 'pending', 'scouting', 'awaiting_ui',
    'cloning', 'filtering', 'submodules', 'manifesting',
    'ready', 'stale', 'failed', 'inaccessible'
);

CREATE TYPE structure_type   AS ENUM ('flat', 'monorepo', 'has_submodules', 'mono_with_submodules');
CREATE TYPE staleness_type   AS ENUM ('code_only', 'structural');
---create table 
CREATE TABLE repos (
    id                  UUID        PRIMARY KEY DEFAULT uuidv7(),
    account_id          UUID        REFERENCES accounts(id) ON DELETE CASCADE,
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
    auto_sync_enabled   BOOL NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


--- create policy 

ALTER TABLE repos ENABLE ROW LEVEL SECURITY,FORCE ROW LEVEL SECURITY;

CREATE POLICY repos_tenant_isolation ON repos
FOR ALL TO fastapi_app_user
USING(
    account_id = current_setting('app.current_account_id',true)::uuid
    OR 
    current_setting('app.is_system_flow',true)='true'
);

-- migrate:down

ALTER TABLE repos DISABLE ROW LEVEL SECURITY,NO FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS repos_tenant_isolation;
DROP TABLE IF EXISTS repos;
DROP EXTENSION IF EXISTS citext;
DROP TYPE IF EXISTS repo_status;
DROP TYPE IF EXISTS structure_type;
DROP TYPE IF EXISTS staleness_type;



