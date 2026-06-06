-- migrate:up

-- Central billing entity. Personal user = one account. Org = one account.
-- Everything downstream references account_id, not user_id directly.

--custom enum types 
CREATE TYPE account_type AS ENUM ('personal','org','enterprise');
CREATE TYPE plan_tier AS ENUM ('free','pro','org','enterprise');

--create table
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT uuidv7(),
    types account_type NOT NULL DEFAULT 'personal',
    plan plan_tier NOT NULL DEFAULT 'free',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updataed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()

);

-- migrate:down

-- drop tables
DROP TABLE IF EXISTS accounts;

-- drop types
DROP TYPES IF EXISTS account_type;
DROP TYPES IF EXISTS plan_tier;
