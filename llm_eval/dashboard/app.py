import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval.adapters import OllamaAdapter  # noqa: E402
from llm_eval.runner.dataset_loader import EvalSample, load_dataset  # noqa: E402
from llm_eval.runner.prompt_runner import PromptRunner  # noqa: E402
from llm_eval.runner.summary import summarize_by_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="LLM Eval Harness", page_icon="⚡", layout="wide")

st.title("LLM Evaluation Harness")
st.caption(
    "Compare local open-source models on accuracy (ROUGE, unit tests), grounding "
    "(sentence-level NLI) and latency. Runs entirely on your machine via Ollama."
)

AVAILABLE_MODELS = ["llama3.1:70b", "deepseek-r1:14b", "phi4", "gemma3:4b"]

DATASETS = {
    "Legal QA": REPO_ROOT / "datasets/legal_qa",
    "Code generation": REPO_ROOT / "datasets/code_gen",
    "Summarization": REPO_ROOT / "datasets/summarization",
}

with st.sidebar:
    st.header("Models")
    selected_models = st.multiselect(
        "Select models to compare",
        AVAILABLE_MODELS,
        default=["gemma3:4b"],
        help="All models run locally via Ollama — pull them first with `ollama pull`.",
    )
    st.caption("Models are queried one at a time so latencies are comparable.")
    st.divider()
    eval_mode = st.radio("Mode", ["Custom question", "Preset benchmark"])

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()


def run_eval(samples, models, use_mlflow=False):
    runner = PromptRunner([OllamaAdapter(m) for m in models], use_mlflow=use_mlflow)
    return asyncio.run(runner.run(samples))


def show_failures(outcome):
    if outcome.failures:
        st.error(outcome.failure_summary())


def fmt_flag(value):
    return "n/a" if value is None or pd.isna(value) else ("flagged" if value else "clean")


if eval_mode == "Custom question":
    st.subheader("Evaluate a custom question")
    user_prompt = st.text_area("Your question", height=100,
                               placeholder="e.g. What is retrieval-augmented generation?")
    expected = st.text_area(
        "Reference answer (used for scoring)",
        height=80,
        placeholder="A short correct answer. ROUGE compares wording with it; the NLI check "
                    "tests whether each sentence of the response is supported by it.",
    )
    concise = st.checkbox("Ask for a short answer (recommended for ROUGE)", value=True)

    if st.button("Evaluate", type="primary"):
        if not user_prompt or not expected:
            st.error("Please fill in both the question and the reference answer.")
            st.stop()

        sample = EvalSample(
            id="custom_001", prompt=user_prompt, expected=expected, domain="custom", tags=["custom"],
            system_prompt="Answer in one or two plain sentences." if concise else "",
        )
        with st.spinner(f"Running {len(selected_models)} model(s)..."):
            outcome = run_eval([sample], selected_models)
        show_failures(outcome)
        if not outcome.results:
            st.stop()

        df = pd.DataFrame([asdict(r) for r in outcome.results])
        display = df[["model_name", "rouge1", "rougeL", "entailment_rate",
                      "hallucination_flag", "latency_ms"]].copy()
        display["hallucination_flag"] = display["hallucination_flag"].map(fmt_flag)
        display.columns = ["Model", "ROUGE-1", "ROUGE-L", "Supported sentences",
                           "Contradiction", "Latency (ms)"]
        st.subheader("Scores")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("Responses")
        for _, row in df.iterrows():
            with st.expander(f"{row['model_name']}  ·  ROUGE-L {row['rougeL']}  ·  "
                             f"contradiction: {fmt_flag(row['hallucination_flag'])}"):
                st.write(row["response_text"])

        st.download_button("Download results CSV", df.to_csv(index=False), "custom_eval.csv", "text/csv")

else:
    st.subheader("Benchmark against a preset dataset")
    dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))
    dataset_path = DATASETS[dataset_label]

    _, col2 = st.columns([3, 1])
    with col2:
        run_bench_btn = st.button("Run benchmark", type="primary", use_container_width=True)

    if run_bench_btn:
        try:
            samples = load_dataset(str(dataset_path))
        except Exception as e:
            st.error(f"Could not load dataset from {dataset_path}: {e}")
            st.stop()

        st.info(f"Running {len(samples)} samples × {len(selected_models)} model(s). "
                f"Large models (70B) may take several minutes per sample on CPU.")

        with st.spinner("Running benchmark..."):
            outcome = run_eval(samples, selected_models, use_mlflow=True)
        show_failures(outcome)

        rows = [asdict(r) for r in outcome.results]
        if not rows:
            st.error("No results returned — check that Ollama is running and the models are pulled.")
            st.stop()
        df = pd.DataFrame(rows)

        st.subheader("Aggregate scores by model")
        agg = pd.DataFrame(summarize_by_model(rows)).T.reset_index().rename(columns={"index": "model"})
        agg = agg.dropna(axis=1, how="all")  # drop metrics that don't apply to this dataset
        st.dataframe(agg, use_container_width=True, hide_index=True)

        headline = "pass_rate" if "pass_rate" in agg else "rougeL_mean"
        st.subheader("Pass rate (unit tests)" if headline == "pass_rate" else "ROUGE-L comparison")
        st.bar_chart(agg.set_index("model")[[headline]])

        st.subheader("Latency comparison (p50, ms)")
        st.bar_chart(agg.set_index("model")[["latency_p50"]])

        if "hallucination_flag" in df and df["hallucination_flag"].notna().any():
            st.subheader("Contradicted responses")
            failures = df[df["hallucination_flag"] == True]  # noqa: E712
            if failures.empty:
                st.success("No response contained a sentence contradicted by its premise.")
            for _, row in failures.iterrows():
                with st.expander(f"[{row['model_name']}] {row['sample_id']}"):
                    st.write("**Prompt:**", row["prompt"])
                    st.write("**Expected:**", row["expected"])
                    st.write("**Got:**", row["response_text"])

        if "tests_passed" in df and df["tests_passed"].notna().any():
            st.subheader("Failed unit tests")
            failed = df[df["tests_passed"] == False]  # noqa: E712
            if failed.empty:
                st.success("Every generated solution passed its tests.")
            for _, row in failed.iterrows():
                with st.expander(f"[{row['model_name']}] {row['sample_id']} — {row['exec_error']}"):
                    st.code(row["response_text"])

        st.subheader("All results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download full results CSV", df.to_csv(index=False),
            f"{dataset_label.lower().replace(' ', '_')}_results.csv", "text/csv",
        )
