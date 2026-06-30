-- migrate:up
-- 1. Drop the restrictive policy
DROP POLICY strict_user_policy ON users;

-- 2. Re-create it to allow 'system flow' to bypass the account check
CREATE POLICY strict_user_policy ON users
FOR ALL TO fastapi_app_user
USING (
    current_setting('app.is_system_flow', true) = 'true'
    OR account_id = current_setting('app.current_account_id', true)::uuid
)
WITH CHECK (
    current_setting('app.is_system_flow', true) = 'true'
    OR account_id = current_setting('app.current_account_id', true)::uuid
);


-- migrate:down
DROP POLICY strict_user_policy ON users;
