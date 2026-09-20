from typing import Any

import numpy as np

# See https://github.com/google-research/google-research/tree/master/rouge
from rouge_score import rouge_scorer


def rouge_eval(summary, ground_truth, threshold=0.3) -> float:
    """
    Evaluate summary using ROUGE scores.

    Args:
    summary (str): The summary to evaluate.
    ground_truth (str): The ground_truth summary.
    threshold (float): The threshold for the ROUGE score (default: 0.3).

    Returns:
    bool: True if the average ROUGE score is above the threshold, False otherwise.
    """

    # rouge1：单个词的重合
    # rouge2：相邻两词的重合
    # rougeL：最长公共子序列的重合, L 指 Longest Common Subsequence
    # use_stemmer=True 启用词干提取
    # 词干提取： 把词的各种变形削回词根，让 terminates、terminated、terminating 算作同一个词
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    # rouge算出来的score，取值范围是0到1，越高表示摘要与参考摘要越相似
    scores = scorer.score(summary, ground_truth)

    print("👍 rouge: " + " ".join(f"{k}={v.fmeasure:.3f}" for k, v in scores.items()))

    # Calculate average ROUGE score
    avg_rouge = np.mean(
        [scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure]
    )

    return float(avg_rouge)


# See https://www.promptfoo.dev/docs/configuration/expected-outputs/python/#external-py
def get_assert(output: str, context, threshold=0.3) -> bool | float | dict[str, Any]:
    ground_truth = context["vars"]["ground_truth"]
    score = rouge_eval(output, ground_truth)

    if score >= threshold:
        return {"pass": True, "score": score, "reason": "Average score is above threshold"}
    else:
        return {"pass": False, "score": score, "reason": "Average score is below threshold"}
