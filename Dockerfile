FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen
COPY . .
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "llm_eval/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
