import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer
from langchain_core.messages import HumanMessage

from agents.FAQ import build_vectorstore, graph, parse_faq_pairs


def evaluate(save_path: Path | None = None):
    build_vectorstore()
    pairs = parse_faq_pairs()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []
    total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for idx, (faq_no, question, reference) in enumerate(pairs):
        state = {
            "messages": [HumanMessage(content=question)],
            "user_question": question,
            "attempt": 0,
            "rewritten_question": "",
        }
        response = graph.invoke(state, config={"configurable": {"thread_id": f"faq-eval-{idx}"}})
        prediction = response.get("messages", [])[-1].content if response.get("messages") else ""

        scores = scorer.score(reference, prediction)
        results.append(
            {
                "faq_no": faq_no,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
        total["rouge1"] += scores["rouge1"].fmeasure
        total["rouge2"] += scores["rouge2"].fmeasure
        total["rougeL"] += scores["rougeL"].fmeasure

    n = len(results) or 1
    averages = {k: v / n for k, v in total.items()}
    print("Average ROUGE scores on FAQ set:")
    for k, v in averages.items():
        print(f"  {k}: {v:.4f}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps({"averages": averages, "items": results}, indent=2), encoding="utf-8")
        print(f"Saved detailed results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FAQ agent with ROUGE against reference FAQ answers.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save per-item results as JSON.")
    args = parser.parse_args()
    evaluate(args.output)
