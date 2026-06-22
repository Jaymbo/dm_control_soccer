-- Initial SQL script for Optuna PostgreSQL database
-- This script runs automatically when the PostgreSQL container is first created.

-- Create extension for UUID (used by Optuna)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant all privileges to the Optuna user (already created by POSTGRES_USER env)
-- This ensures Optuna can create tables and indexes

-- No additional setup required - Optuna creates tables automatically on first run.
-- This script is here for future extensions or custom configurations.

-- Optional: Create a read-only user for dashboard-only access
-- CREATE USER optuna_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE optuna_db TO optuna_readonly;
-- GRANT USAGE ON SCHEMA public TO optuna_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO optuna_readonly;
