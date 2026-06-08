-- migrate:up
USERS TABLE (Identity & Auth)

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    github_id BIGINT UNIQUE NOT NULL,
    github_login VARCHAR(100) NOT NULL,
    github_email VARCHAR(255),
    role VARCHAR(20) NOT NULL DEFAULT 'owner',    -- 'owner', 'admin', 'member'
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- migrate:down

DROP TABLE IF EXISTS users