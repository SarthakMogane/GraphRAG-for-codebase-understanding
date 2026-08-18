-- migrate:up
ALTER TABLE repos 
ADD COLUMN IF NOT EXISTS using_git_lfs BOOLEAN;

-- migrate:down
ALTER TABLE repos 
DROP COLUMN IF EXISTS using_git_lfs;

