-- migrate:up

CREATE TABLE webhooks_received (
    id                  UUID        PRIMARY KEY DEFAULT uuidv7(),
    delivery_id         TEXT        NOT NULL UNIQUE,            -- X-GitHub-Delivery header
    event_type          TEXT  NOT NULL,
    github_install_id   BIGINT      NOT NULL,
    repo_full_name      TEXT,
    payload             JSONB       NOT NULL,

    -- Processing state
    processed           BOOL        NOT NULL DEFAULT FALSE,
    processed_at        TIMESTAMPTZ,
    error               TEXT,                                   -- if processing failed

    received_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

--policy
ALTER TABLE webhooks_received ENABLE ROW LEVEL SECURITY;

CREATE POLICY webhook_isolation on webhooks_received
FOR ALL TO fastapi_app_user
USING (current_setting('app.is_system_flow',true)='true')
WITH CHECK (
-- Ensure writes/inserts are also strictly bound to the system context
        current_setting('app.is_system_flow', true) = 'true'
    );

-- migrate:down
ALTER TABLE DISABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS webhook_isolation;
DROP TABLE IF EXISTS webhooks_received;
