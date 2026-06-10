-- migrate:up

---accounts TABLE POLICY

CREATE POLICY account_isolation_policy ON accounts
    FOR ALL TO fastapi_app_user
    USING (id = current_settings('app.current_account_id',true)::uuid)
    WITH CHECK (id = current_settings('app.current_account_id')::uuid);
    ----Notice that we are matching the table's id column to the account_id nametag, because this is the root table.

CREATE POLICY acc_auth_insert ON accounts
    FOR INSERT TO fastapi_app_user
    WITH CHECK (current_settings('app.is_auth_flow',true)='true');


---USER TABLE POLICY 
CREATE POLICY strict_user_policy ON users
    FOR ALL TO fastapi_app_user
    USING (id = current_settings('app.current_account_id')::uuid)
    WITH CHECK (id = current_settings('app.current_account_id')::uuid);

CREATE POLICY user_auth_select ON users 
    FOR SELECT TO fastapi_app_user
    USING (current_settings('app.is_auth_flow',true)= 'true');

CREATE POLICY user_auth_insert ON users 
    FOR INSERT TO fastapi_app_user
    WITH CHECK (current_settings('app.is_auth_flow',true)= 'true');

-- migrate:down
DROP POLICY IF EXISTS account_isolation_policy ON accounts;
DROP POLICY IF EXISTS acc_auth_insert ON accounts;

DROP POLICY IF EXISTS strict_user_policy ON users;
DROP POLICY IF EXISTS user_auth_select ON users;
DROP POLICY IF EXISTS user_auth_insert ON users;




