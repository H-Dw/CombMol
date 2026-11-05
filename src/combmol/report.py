from pathlib import Path


def formatted(value):
    return "未测量" if value is None else f"{value:.4f}"


def campaign_report(root, report):
    lines = ["# CombMol 双靶点融合生成报告", "",
             f"协议：`{report['protocol']}`；提交：`{report['revision']}`。",
             f"实际开始的科学尝试：{report['spent_attempts']}/{report['upper_bound']}。",
             "使用原生 fragment-growing inpainting，无梯度引导，终态不强制覆盖条件。", "",
             "| 靶点 | 融合方案 | 尝试 | 有效融合 | 有效融合 mean pIC50 | 有效融合 best pIC50 | 父本 pIC50 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for target in report["targets"]:
        if target.get("status") == "failed":
            lines += [f"| {target['target']} | 任务失败，详见失败回执 | — | — | — | — | — |"]
        for position, summary in enumerate(target.get("summaries", []), 1):
            lines.append(f"| {target['target']} | fusion-{position:02d} | {summary['attempts']} | "
                         f"{summary['successful_fusions']} | {formatted(summary['valid_fusion_mean_pic50'])} | "
                         f"{formatted(summary['valid_fusion_best_pic50'])} | " + ", ".join(formatted(value) for value in summary["parent_pic50"]) + " |")
        if target.get("status") == "no_feasible_combination":
            lines += [f"| {target['target']} | 无可行组合；未补齐或额外生成 | 0 | 0 | — | — | — |"]
    lines += ["", "有效融合要求两侧条件图、蛋白坐标系中的位置及至少一个预期互作均保留，并满足连通性和 clash 门槛。",
              "均值和最佳值仅针对有效融合，同时保留全部尝试及失败分母。预测亲和力变化不等同于实验活性变化。",
              "逐候选结果见各靶点和融合方案的 candidates.csv；筛选、LLM请求及来源信息均有独立回执。", ""]
    Path(root, "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
