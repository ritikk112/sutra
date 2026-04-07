-- Load Apache AGE (graph storage, openCypher queries)
CREATE EXTENSION IF NOT EXISTS age;

-- Load pgvector (embedding storage)
CREATE EXTENSION IF NOT EXISTS vector;

-- Make AGE catalog visible in every session without manual SET search_path
ALTER DATABASE postgres SET search_path = ag_catalog, "$user", public;
