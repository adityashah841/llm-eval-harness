FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY llm_eval ./llm_eval
COPY api ./api
COPY datasets ./datasets
# Runtime state lives outside the image: mount a volume at /app/data to keep
# run history, and at /root/.cache/huggingface to avoid re-downloading the
# NLI model on every start.
ENV LLM_EVAL_DB_PATH=/app/data/metrics.db
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
