-- migrate:up
CREATE TYPE  CloneStrategy AS ENUM ('shallow','partial_blob','sparse_checkout');

ALTER TABLE repos
ALTER COLUMN clone_strategy TYPE CloneStrategy
USING clone_strategy::CloneStrategy;

-- migrate:down
ALTER TABLE repos 
ALTER COLUMN clone_strategy TYPE TEXT;

DROP TYPE CloneStrategy;