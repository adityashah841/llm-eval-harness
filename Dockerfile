FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY llm_eval ./llm_eval
COPY datasets ./datasets
EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "llm_eval/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
