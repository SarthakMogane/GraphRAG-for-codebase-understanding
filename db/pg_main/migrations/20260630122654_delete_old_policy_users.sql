-- migrate:up
DROP POLICY IF EXISTS user_auth_insert ON users;
DROP POLICY IF EXISTS user_auth_select ON users;

-- migrate:down

CREATE POLICY user_auth_select ON users
FOR SELECT TO fastapi_app_user
USING (current_setting('app.is_system_flow', true) = 'true');

CREATE POLICY user_auth_insert ON users
FOR INSERT TO fastapi_app_user
WITH CHECK (current_setting('app.is_system_flow', true) = 'true');