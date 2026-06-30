-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS github_name TEXT;

-- migrate:down

ALTER TABLE users DROP COLUMN IF EXISTS github_name;