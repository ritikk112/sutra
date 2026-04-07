FROM postgres:16

# Install build tools
# flex + bison: required by Apache AGE to generate its lexer/parser (ag_scanner.l → ag_scanner.c)
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        flex \
        bison \
        git \
        ca-certificates \
        postgresql-server-dev-16 && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Apache AGE
# Tag format: PG{pg_major}/v{age_version} — verified from https://github.com/apache/age/tags
RUN git clone --branch PG16/v1.6.0-rc0 --depth 1 https://github.com/apache/age.git && \
    cd age && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config && \
    make install && \
    cd .. && rm -rf age

# Install pgvector
RUN git clone --branch v0.7.4 --depth 1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config && \
    make install && \
    cd .. && rm -rf pgvector

# AGE must be loaded at server start via shared_preload_libraries
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample

# Auto-create extensions on first `docker run`
# This script runs inside the default database ($POSTGRES_DB) on first startup
COPY docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/
