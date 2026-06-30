-- migrate:up
DROP POLICY IF EXISTS auth_settings_manage ON user_settings;

CREATE POLICY settings_isolation ON user_settings
FOR ALL TO fastapi_app_user
USING (
    current_setting('app.is_system_flow',true)= 'true'
)
WITH CHECK(
    current_setting('app.is_system_flow',true)= 'true');

-- migrate:down

DROP POLICY IF EXISTS settings_isolation ON user_settings;
