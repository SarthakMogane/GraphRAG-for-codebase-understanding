-- migrate:up
-- 1. Create the restricted user for your local FastAPI app
CREATE USER fastapi_app_user WITH PASSWORD '3kP9!vX$mQ2_zL7aRt';

-- 2. Give them basic read/write access, but NOT permission to drop/alter tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fastapi_app_user;

-- 3. (Important for future migrations) Tell Postgres to automatically grant 
-- these same permissions to any brand NEW tables you create in the future
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fastapi_app_user;

-- migrate:down
-- 1. Revoke the automatic default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM fastapi_app_user;

-- 2. Revoke permissions on all existing tables
REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM fastapi_app_user;

-- 3. Drop the restricted user
DROP USER fastapi_app_user;


