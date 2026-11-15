docker run -d \
  --name postgres-vector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres123 \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -v postgres_vector_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  pgvector/pgvector:pg16
