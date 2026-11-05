"""Render the audited 20261106 CombMol campaign and its original top-hit structures."""
import argparse
import collections
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path


def number(value):
    return "—" if value is None or value == "" else f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    options = parser.parse_args()
    folder = Path(options.folder)
    analysis = json.loads((folder / "analysis.json").read_text())
    assert analysis["scientific_revision"] == "fa1d2a258ba38aeb9466545c1b5de217b8d1e27b" and analysis["attempts"] == 675
    storage = json.loads((folder / "storage_audit.json").read_text())
    assert analysis["passed"] and storage["passed"]
    load_csv = lambda name: list(csv.DictReader((folder / name).open(encoding="utf-8-sig")))
    candidates, parents, groups = [load_csv(name) for name in ("candidate_metrics.csv", "parent_metrics.csv", "fusion_summary.csv")]
    with tarfile.open(folder / "evidence.tar.gz", "r:gz") as stream:
        files = {item.name: stream.extractfile(item).read() for item in stream if item.isfile()}
    read = lambda name: json.loads(files[name])
    aliases = {row["identifier"]: row["parent"] for row in parents}
    quality = {}
    origins = {}
    structures = folder / "top_structures"
    structures.mkdir(exist_ok=True)

    def export(name, molblock, properties, source):
        contents = molblock.rstrip() + "\n\n" + "".join(f"> <{key}>\n{value}\n\n" for key, value in properties.items()) + "$$$$\n"
        path = structures / name
        path.write_text(contents, encoding="utf-8", newline="\n")
        origins[name] = dict(source=source, properties=properties, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    lines = ["# CombMol 双靶点原生 inpainting：完成结果与独立核验", "",
        f"科学运行固定提交：`{analysis['scientific_revision']}`，最终尝试 `attempt_003`。",
        "两靶点的采样和全部CPU诊断均已完成。27个方案各25次，共675/800次；未补齐125次剩余额度。",
        "6pyy由LLM选择11个方案，3w54选择16个；各方案使用相同五种子，每种子五个候选，批量上限50。", "",
        "合格融合沿用冻结协议：连通、固定区域化学保留、两侧条件RMSD均严格<0.5Å、无严重蛋白clash、双方至少一个预期PLIP互作恢复。",
        "父本共用原始蛋白坐标系，对齐为恒等变换；没有再单独拟合配体。PoseBusters（PB）仅诊断，PB不通过的候选仍可能满足该合格融合定义。", "",
        "| 靶点 | 方案 | 全部尝试 | 可构建 | 合格融合 | 合格率 | 合格pIC50均值 | Top10%均值 | Top1 | 最佳父本 | Top1−最佳父本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for target in analysis["targets"]:
        lines.append(f"| {target['target'].split('_')[0]} | {target['conditions']} | {target['attempts']} | {target['built']} | {target['qualified']} | {100*target['qualified_rate']:.2f}% | {number(target['qualified_pic50_mean'])} | {number(target['qualified_pic50_top10_mean'])} | {number(target['qualified_pic50_best'])} | {number(target['best_parent_pic50'])} | {target['best_delta_over_best_parent']:+.4f} |")
    lines += ["", "Top10%取合格融合中按pIC50排序的前ceil(0.1×数量)个，因此分别为8/75和2/17个。所有均值按候选加权，不是方案均值的再次平均。",
        "182次构建失败仍保留在675次尝试分母内；其pIC50缺失，不能赋成零或算入有评分的分子均值。", "",
        "**pIC50、SA和QED的集合比较**", "",
        "下表Top10%和Top1均按pIC50选择，再读取同一集合的SA/QED。SA低值代表较易合成的模型估计；这些指标不构成实际活性或合成验证。", "",
        "| 靶点 | 集合 | 数量 | mean pIC50 | mean SA | mean QED | mean重原子数 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for target in analysis["targets"]:
        short = target["target"].split("_")[0]
        parent_rows = [row for row in parents if row["target"] == target["target"]]
        lines.append(f"| {short} | 所选父本 | 8 | {statistics.mean(float(row['pic50']) for row in parent_rows):.4f} | {statistics.mean(float(row['SAScore']) for row in parent_rows):.4f} | {statistics.mean(float(row['QEDScore']) for row in parent_rows):.4f} | {statistics.mean(float(row['heavy_atoms']) for row in parent_rows):.2f} |")
        for label, key, count in [("全部可构建", "built", target["built"]), ("合格融合", "qualified", target["qualified"]), ("合格Top10%", "qualified_top10_by_pic50", target["qualified_top10_count"])]:
            lines.append(f"| {short} | {label} | {count} | {number(target[key+'_pic50_mean'])} | {number(target[key+'_SAScore_mean'])} | {number(target[key+'_QEDScore_mean'])} | {number(target[key+'_heavy_atoms_mean'])} |")
        best = next(row for row in candidates if row["target"] == target["target"] and row["node_id"] == target["best_qualified_node"])
        lines.append(f"| {short} | 合格Top1 | 1 | {number(best['pic50'])} | {number(best['SAScore'])} | {number(best['QEDScore'])} | {best['heavy_atoms']} |")
        built = [row for row in candidates if row["target"] == target["target"] and row["build_success"] == "True"]
        qualified = [row for row in built if row["fusion_success"] == "True"]
        pb_pass = [row for row in qualified if row["posebusters_pass"] == "True"]
        measure = read(best["evaluation_path"])
        quality[short] = dict(build_failures=target["attempts"] - len(built),
            disconnected=sum(row["connected"] == "False" for row in built),
            changed_condition_chemistry=sum(row["chemistry_retained"] == "False" for row in built),
            rmsd_at_least_0_5=sum(float(row["max_condition_rmsd_A"]) >= .5 for row in built),
            protein_clash=sum(row["protein_clash"] == "True" for row in built),
            missing_either_parent_contact=sum(row["both_contacts_recovered"] == "False" for row in built),
            qualified_PB_pass=len(pb_pass), parent_PB_pass=sum(row["posebusters_pass"] == "True" for row in parent_rows),
            best_PB_failed_checks=[name for name, passed in measure["posebusters"].get("posebusters_checks", {}).items() if not passed],
            best_PB_pass_qualified_pic50=max(float(row["pic50"]) for row in pb_pass) if pb_pass else None,
            qualified_above_both_parents_and_PB_pass=sum(row["exceeds_both_parents"] == "True" for row in pb_pass))
        for label, candidate in [("best_qualified", best), ("best_PB_pass", max(pb_pass, key=lambda row: float(row["pic50"])) if pb_pass else None)]:
            if candidate is not None:
                records_path = candidate["evaluation_path"].rsplit("/", 2)[0] + "/TERMINAL_RECORDS.json"
                terminal = next(row for row in read(records_path) if row["node_id"] == candidate["node_id"])
                export(short + "_" + label + ".sdf", terminal["molblock"], dict(node_id=candidate["node_id"], pic50=candidate["pic50"], SAScore=candidate["SAScore"], posebusters_pass=candidate["posebusters_pass"]), records_path)
        condition = read("targets/" + target["target"] + "/selection/CONDITIONS.json")
        winning = next(row for row in condition if row["proposal"]["identifier"] == best["fusion"])
        parent_packet = read("targets/" + target["target"] + "/selection/PARENTS.json")["parents"]
        for position, contribution in enumerate(winning["proposal"]["parent_regions"], 1):
            parent = next(row for row in parent_packet if row["identifier"] == contribution["parent_identifier"])
            export(short + f"_winning_parent_{position}.sdf", parent["molblock"], dict(node_id=parent["identifier"], pic50=parent["pic50"], evidence_id=parent["evidence_id"]), "targets/" + target["target"] + "/selection/PARENTS.json")
    lines += ["", "**结果的实际含义**", "",
        "6pyy的75个合格融合只对应16种不同拓扑。仅3个候选超过各自双方父本；全局最佳为fusion-11_seed100045_rep4，pIC50=5.1493，比八个父本中的最高分高0.4567，且通过PB。该方案仅1/25合格，因此这是稀疏的最佳样本收益；总体合格均值3.7201低于全部所选父本，不能据此宣称平均亲和力提升。", "",
        "3w54的17个合格融合对应9种不同拓扑。3个超过各自双方父本，但没有候选超过八个父本中的最高分7.4919。最佳7.4714不通过PB的bond_lengths和bond_angles检查；17个合格融合仅1个PB全部通过，该候选pIC50=6.3446。按冻结定义合格，不等于整体物理几何合格。", "",
        "化学筛选分别保留92/98与95/97个可构建来源姿态，再按评分和拓扑去重选8个父本。所选父本PB通过数分别8/8和6/8。当前证据证明流程能够产生保留双方局部互作的融合结构；尚未通过等预算随机配对或无LLM对照证明LLM决策的额外收益。", "",
        "**未通过条件的统计**", "",
        "下列原因只统计可构建终态，原因之间可重叠，不能相加当作失败总数。", "",
        "| 靶点 | 构建失败 | 不连通 | 条件化学改变 | 任一侧RMSD≥0.5Å | 严重蛋白clash | 任一侧预期互作未恢复 | 合格融合中PB通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for short, item in quality.items():
        lines.append(f"| {short} | {item['build_failures']} | {item['disconnected']} | {item['changed_condition_chemistry']} | {item['rmsd_at_least_0_5']} | {item['protein_clash']} | {item['missing_either_parent_contact']} | {item['qualified_PB_pass']} |")
    lines += ["", "主要筛除项是终态条件位置偏离：6pyy为162/241、3w54为229/252个可构建结构。6pyy全部可构建结构仍有双方预期互作，说明PLIP标签恢复与严格位置保持是不同指标。3w54还存在较多条件化学改变、蛋白clash和连接问题。终态未强制覆盖条件，所有偏差均在实际返回结构上测量。", "",
        "**逐方案表**", "", "parent编号按该靶点父本pIC50排序，完整node_id见parent_metrics.csv。", "",
        "| 靶点 | 方案 | 父本 | 可构建/25 | 合格/25 | 合格mean pIC50 | 合格Top1 | 两父本pIC50 | Top1−较好父本 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for group in groups:
        delta = "—" if not group["best_delta_over_better_parent"] else f"{float(group['best_delta_over_better_parent']):+.4f}"
        lines.append(f"| {group['target'].split('_')[0]} | {group['fusion']} | {aliases[group['parent_left']]} + {aliases[group['parent_right']]} | {group['built']} | {group['qualified']} | {number(group['mean_pic50'])} | {number(group['best_pic50'])} | {number(group['parent_left_pic50'])}, {number(group['parent_right_pic50'])} | {delta} |")
    lines += ["", "**审计与交付**", "",
        "675个唯一槽全部有终态记录及评价，27个方案各25条，135个批次各5条，无预留失败槽或评价缺失。对493个可构建分子独立重算条件RMSD、化学保留、连通性、蛋白clash和双方接触恢复，与原回执一致；RMSD最大差为0。",
        "已验证原始数据归档、checkpoint、234个冻结上游文件及19个恢复前已完成选择的哈希。135批初始化和终态张量均校验；SDF坐标与原始终态的最大差为0.00005Å，来自四位小数序列化。",
        "135批均有五个不同的完整初始张量输入。三批只有一个自由原子，原生零质心先验使该原子的坐标相同，但分类特征不同；没有把坐标相同误判为重复采样。",
        "失败尝试attempt_001/002在生成前停止，科学消耗均为0；其错误、请求和修复谱系仍保留。attempt_003完成全部675次。主目录历史FAILED文件属于旧尝试，最终状态以最新尝试和独立审计为准。", "",
        "[675条候选表](candidate_metrics.csv) · [27个方案表](fusion_summary.csv) · [16个父本表](parent_metrics.csv) · [两靶点汇总](target_summary.csv) · [独立审计](analysis.json) · [原始文件与张量审计](storage_audit.json) · [原始小体积证据包](evidence.tar.gz) · [最佳结构与父本](top_structures/ORIGINS.json)", "",
        "完整原始张量继续保留在远端/workspace；此证据包不包含checkpoint或全量中间张量。预测pIC50和结构筛选不替代实验测量。", ""]
    (folder / "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (folder / "quality_audit.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (structures / "ORIGINS.json").write_text(json.dumps(origins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(targets=2, candidate_rows=len(candidates), schemes=len(groups), structure_files=len(origins), quality=quality)))


if __name__ == "__main__":
    main()
