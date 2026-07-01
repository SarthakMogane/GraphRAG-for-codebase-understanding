-- migrate:up
DROP FUNCTION IF EXISTS get_auth_status(UUID, UUID);

CREATE OR REPLACE FUNCTION get_auth_status(p_account_id UUID, p_user_id UUID)
RETURNS TABLE (
    id UUID,
    github_login VARCHAR(100),
    avatar_url TEXT,
    plan plan_tier,
    is_installed BOOLEAN
) AS $$
BEGIN
    -- 1. 🛡️ This is guaranteed to execute FIRST in memory
    PERFORM set_config('app.current_account_id', p_account_id::text, true);
    
    -- 2. Fetch the records natively (RLS reads the config perfectly now!)
    RETURN QUERY
    SELECT
        u.id,
        u.github_login,
        u.avatar_url,
        a.plan,
        EXISTS(
            SELECT 1 FROM installations i
            WHERE i.account_id = a.id
                AND i.is_active = TRUE
        ) AS is_installed
    FROM users u
    JOIN accounts a ON a.id = u.account_id
    WHERE u.id = p_user_id
      AND u.account_id = p_account_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER; 

-- migrate:down

CREATE OR REPLACE FUNCTION get_auth_status(p_account_id UUID, p_user_id UUID)
RETURNS TABLE (
    id UUID,
    github_login TEXT,
    avatar_url TEXT,
    plan TEXT,
    is_installed BOOLEAN
) AS $$
BEGIN
    -- 1. 🛡️ This is guaranteed to execute FIRST in memory
    PERFORM set_config('app.current_account_id', p_account_id::text, true);
    
    -- 2. Fetch the records natively (RLS reads the config perfectly now!)
    RETURN QUERY
    SELECT
        u.id,
        u.github_login,
        u.avatar_url,
        a.plan,
        EXISTS(
            SELECT 1 FROM installations i
            WHERE i.account_id = a.id
                AND i.is_active = TRUE
        ) AS is_installed
    FROM users u
    JOIN accounts a ON a.id = u.account_id
    WHERE u.id = p_user_id
      AND u.account_id = p_account_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER; 
