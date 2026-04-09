import streamlit as st
import pandas as pd
import asyncio
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval.adapters import OllamaAdapter
from llm_eval.runner.prompt_runner import PromptRunner
from llm_eval.runner.dataset_loader import load_dataset, EvalSample

st.set_page_config(
    page_title="LLM Eval Harness",
    page_icon="⚡",
    layout="wide",
)

st.title("LLM Evaluation Harness")
st.caption(
    "Compare local open-source models across accuracy, hallucination rate, "
    "and latency. Runs entirely on your machine via Ollama — zero API keys."
)

AVAILABLE_MODELS = [
    "llama3.1:70b",
    "deepseek-r1:14b",
    "phi4",
    "gemma3:4b",
]

DATASETS = {
    "Legal QA": "datasets/legal_qa",
    "Code generation": "datasets/code_gen",
    "Summarization": "datasets/summarization",
}

with st.sidebar:
    st.header("Models")
    selected_models = st.multiselect(
        "Select models to compare",
        AVAILABLE_MODELS,
        default=["gemma3:4b", "phi4"],
        help="All models run locally via Ollama — no API keys needed.",
    )
    st.caption("Larger models are slower but more accurate.")

    st.divider()
    eval_mode = st.radio("Mode", ["Custom question", "Preset benchmark"])

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()


def run_eval(samples, models, use_mlflow=False):
    adapters = [OllamaAdapter(m) for m in models]
    runner = PromptRunner(adapters, use_mlflow=use_mlflow)
    return asyncio.run(runner.run(samples))


if eval_mode == "Custom question":
    st.subheader("Evaluate a custom question")
    user_prompt = st.text_area(
        "Your question",
        height=100,
        placeholder="e.g. What is retrieval-augmented generation?",
    )
    expected = st.text_area(
        "Reference answer (used for scoring)",
        height=80,
        placeholder="Paste the correct answer here so ROUGE and hallucination scores can be computed.",
    )
    run_btn = st.button("Evaluate", type="primary")

    if run_btn:
        if not user_prompt or not expected:
            st.error("Please fill in both the question and the reference answer.")
            st.stop()

        sample = EvalSample(
            id="custom_001",
            prompt=user_prompt,
            expected=expected,
            domain="custom",
            tags=["custom"],
        )
        with st.spinner(f"Running across {len(selected_models)} model(s) in parallel..."):
            results = run_eval([sample], selected_models)

        df = pd.DataFrame([asdict(r) for r in results])

        st.subheader("Scores")
        display = df[
            ["model_name", "rouge1", "rouge2", "rougeL",
             "hallucination_flag", "nli_confidence", "latency_ms"]
        ].copy()
        display.columns = [
            "Model", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "Hallucination", "NLI confidence", "Latency (ms)"
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("Responses")
        for _, row in df.iterrows():
            flag = "Hallucination detected" if row["hallucination_flag"] else "No hallucination"
            with st.expander(
                f"{row['model_name']}  ·  ROUGE-L {row['rougeL']}  ·  {flag}"
            ):
                st.write(row["response_text"])

        csv = df.to_csv(index=False)
        st.download_button("Download results CSV", csv, "custom_eval.csv", "text/csv")

else:
    st.subheader("Benchmark against a preset dataset")
    dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))
    dataset_path = DATASETS[dataset_label]

    col1, col2 = st.columns([3, 1])
    with col2:
        run_bench_btn = st.button("Run benchmark", type="primary",
                                   use_container_width=True)

    if run_bench_btn:
        try:
            samples = load_dataset(dataset_path)
        except Exception as e:
            st.error(f"Could not load dataset from {dataset_path}: {e}")
            st.stop()

        st.info(
            f"Running {len(samples)} samples × {len(selected_models)} model(s). "
            f"Large models (70B) may take several minutes per sample."
        )

        with st.spinner("Running benchmark..."):
            results = run_eval(samples, selected_models, use_mlflow=True)

        df = pd.DataFrame([asdict(r) for r in results])

        st.subheader("Aggregate scores by model")
        agg = (
            df.groupby("model_name")
            .agg(
                rouge1_mean=("rouge1", "mean"),
                rougeL_mean=("rougeL", "mean"),
                halluc_rate=("hallucination_flag", "mean"),
                latency_p50=("latency_ms", "median"),
                latency_p95=("latency_ms", lambda x: x.quantile(0.95)),
            )
            .round(4)
            .reset_index()
        )
        agg.columns = [
            "Model", "ROUGE-1 mean", "ROUGE-L mean",
            "Hallucination rate", "Latency p50 (ms)", "Latency p95 (ms)"
        ]
        st.dataframe(agg, use_container_width=True, hide_index=True)

        st.subheader("ROUGE-L comparison")
        chart_data = agg.set_index("Model")[["ROUGE-L mean"]]
        st.bar_chart(chart_data)

        st.subheader("Latency comparison (p50)")
        latency_data = agg.set_index("Model")[["Latency p50 (ms)"]]
        st.bar_chart(latency_data)

        st.subheader("Failure cases (hallucinations detected)")
        failures = df[df["hallucination_flag"]]
        if failures.empty:
            st.success("No hallucinations detected across any model.")
        else:
            st.warning(f"{len(failures)} hallucination(s) detected.")
            for _, row in failures.iterrows():
                with st.expander(f"[{row['model_name']}] {row['sample_id']}"):
                    st.write("**Prompt:**", row["prompt"])
                    st.write("**Expected:**", row["expected"])
                    st.write("**Got:**", row["response_text"])

        st.subheader("All results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "Download full results CSV", csv,
            f"{dataset_label.lower().replace(' ', '_')}_results.csv",
            "text/csv",
        )
