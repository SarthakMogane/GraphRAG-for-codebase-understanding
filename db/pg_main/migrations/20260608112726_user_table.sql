-- migrate:up
CREATE TYPE user_role AS ENUM ('owner','admin','member','viewer');

USERS TABLE (Identity & Auth)

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuidv7(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    github_id BIGINT UNIQUE NOT NULL,
    github_login VARCHAR(100) NOT NULL,
    github_email VARCHAR(255),

    -- OAuth tokens (encrypted at rest via AWS KMS before insert)
    -- Store the encrypted blob, decrypt in application layer
    oauth_token_enc     BYTEA,                                  -- encrypted access token
    oauth_token_expires TIMESTAMPTZ,
    refresh_token_enc   BYTEA,                                  -- encrypted refresh token

    role user_role NOT NULL DEFAULT 'owner'
    avatar_url TEXT,

    -- Profile preferences (stored in user_settings table for extensibility)
    timezone            TEXT        NOT NULL DEFAULT 'UTC',
    locale              TEXT        NOT NULL DEFAULT 'en',

    last_seen_at        TIMESTAMPTZ,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


--RLS 
ALTER users ENABLE ROW LEVEL SECURITY , FORCE ROW LEVEL SECURITY;


-- migrate:down

-- drop RLS 
ALTER users DISABLE ROW LEVEL SECURITY , NO FORCE ROW LEVEL SECURITY;
-- drop tables 
DROP TABLE IF EXISTS users;
--- drop types 
DROP TYPE IF EXISTS user_role;

