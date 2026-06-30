-- migrate:up
DROP POLICY IF EXISTS account_isolation_policy ON accounts;
DROP POLICY IF EXISTS acc_auth_insert ON accounts;

DROP POLICY IF EXISTS strict_user_policy ON users;
DROP POLICY IF EXISTS user_auth_select ON users;
DROP POLICY IF EXISTS user_auth_insert ON users;

ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- users policy 

CREATE POLICY strict_user_policy ON users
FOR ALL TO fastapi_app_user
USING (account_id = current_setting('app.current_account_id',true)::uuid)
WITH CHECK (account_id = current_setting('app.current_account_id',true)::uuid);

CREATE POLICY user_auth_select ON users
FOR SELECT TO fastapi_app_user
USING (current_setting('app.is_system_flow',true)='true');

CREATE POLICY user_auth_insert ON users
FOR INSERT TO fastapi_app_user
WITH CHECK (current_setting('app.is_system_flow',true)='true');


-- accounts policy 

CREATE POLICY account_isolation_policy ON accounts
FOR ALL TO fastapi_app_user
USING (
    id = current_setting('app.current_account_id',true)::uuid
    OR 
    current_setting('app.is_system_flow',true)= 'true'
    )
WITH CHECK (
    id = current_setting('app.current_account_id',true)::uuid
    OR 
    current_setting('app.is_system_flow',true)= 'true'
    );


-- migrate:down

DROP POLICY IF EXISTS strict_user_policy ON users;
DROP POLICY IF EXISTS user_auth_select ON users;
DROP POLICY IF EXISTS user_auth_insert ON users;

DROP POLICY IF EXISTS account_isolation_policy ON accounts;
