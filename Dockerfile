FROM postgres:16

# Install build tools for pgvector
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        ca-certificates \
        postgresql-server-dev-16 && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN git clone --branch v0.7.4 --depth 1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config && \
    make install && \
    cd .. && rm -rf pgvector

# Auto-create extensions on first `docker run`.
# This script runs inside the default database ($POSTGRES_DB) on first startup.
COPY docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/
