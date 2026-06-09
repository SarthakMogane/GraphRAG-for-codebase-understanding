-- migrate:up


-- USER SETTINGS
-- Extensible key-value profile settings per user.
-- One row per user — use JSONB for arbitrary settings without migrations.
-- -----------------------------------------------------------------------------
CREATE TABLE user_settings (
    user_id             UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,

    -- Chat preferences
    default_model       TEXT        DEFAULT NULL,
    response_style      TEXT        NOT NULL DEFAULT 'detailed',   -- 'detailed' | 'concise' | 'bullets'
    show_citations      BOOL        NOT NULL DEFAULT TRUE,
    show_file_context   BOOL        NOT NULL DEFAULT TRUE,
    code_theme          TEXT        NOT NULL DEFAULT 'github-dark',

    -- Notification preferences
    notify_index_done   BOOL        NOT NULL DEFAULT TRUE,
    notify_stale        BOOL        NOT NULL DEFAULT FALSE,
    notify_email        BOOL        NOT NULL DEFAULT FALSE,

    -- UI preferences (arbitrary, stored as JSONB for future extensibility)
    ui_prefs            JSONB       NOT NULL DEFAULT '{}',

    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

---RLS 
ALTER user_settings ENABLE ROW LEVEL SECURITY , FORCE ROW LEVEL SECURITY

--- policy
CREATE POLICY settings_manage ON user_settings
FOR ALL TO fastapi_app_user
USING (id = current_setting('app.current_user_id',true)::uuid)
WITH CHECK ( id = current_setting('app.current_user_id',true)::uuid)


CREATE POLICY auth_settings_manage ON user_settings
FOR INSERT TO fastapi_app_user
WITH CHECK (current_settings('app.is_auth_flow',true)= 'true')



-- migrate:down

DROP TABLE IF EXISTS
DROP POLICY IF EXISTS
DROP POLICY IF EXISTS

ALTER user_settings DISABLE ROW LEVEL SECURITY, NO FORCE ROW LEVEL SECURITY

