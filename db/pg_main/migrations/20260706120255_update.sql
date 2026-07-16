-- migrate:up
DROP POLICY IF EXISTS selections_isolation_policy ON user_selections;

CREATE POLICY selections_isolation_policy ON user_selections
    FOR ALL TO fastapi_app_user
    USING (
        account_id = current_setting('app.current_account_id', true)::uuid
        OR 
        current_setting('app.is_system_flow', true) = 'true'
    )
    WITH CHECK (
        account_id = current_setting('app.current_account_id', true)::uuid  -- Fixed typo here!
        OR 
        current_setting('app.is_system_flow', true) = 'true'
    );

-- migrate:down

CREATE POLICY selections_isolation_policy ON user_selections
    FOR ALL TO fastapi_app_user
    USING (
        account_id = current_setting('app.current_account_id',true)::uuid
        OR 
        current_setting('app.is_system_flow',true) = 'true'
    )
    WITH CHECK (
        account_id = current_setting('app.current_id',true)::uuid
        OR 
        current_setting('app.is_system_flow',true) = 'true'
    );

