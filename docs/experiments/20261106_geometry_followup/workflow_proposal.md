# Agent选区与恢复闭环：下一版设计草案

只提出设计，不修改当前schema、skills、checkpoint、实验配置或已完成结果。

    ParentAnchorAssess:
        程序给出完整父本图、PB、实际互作和预测评分。
        Agent选择主体与候选核心锚点，声明保留和可牺牲的特征。
    RegionContextAssemble:
        程序为每个实际子图生成保留/删除原子、官能团、取代环境和互作清单。
        将母体具有但子图已删除的性质明确标注，避免属性继承错误。
    PortFeasibilityScreen:
        程序列出允许接合的端口、允许封端的位置、价态和键型。
        在固定源坐标和蛋白环境内检查多种连接规模及构象可行性。
        返回有来源的可行性记录，不要求LLM构造完整融合图。
    ConditionNoveltyCluster:
        按实际保留图、关键特征和几何聚类。
        让Agent看到重复条件和真实不同的机会。
    AgentDecide:
        从可行的机会中选择、替换或放弃。
        给出保留特征、放弃特征、互补理由和程序证据的简短摘要。
    NativeInpaint:
        按已冻结的新协议生成完整终态。
        明确最后一次预测是否继续保持固定区域，保持实际返回结构可追踪。
    TerminalValidate:
        对实际返回的完整化学图和坐标评价接缝、PB、互作类型/几何和pIC50。
        同时保留所有尝试及失败分母。
    FeedbackAccept:
        Agent根据实测结果决定保留主体、调整补充区域或退出该分支。
        成功父本和条件可复用；新条件单独登记，不能重用已消耗的旧实验槽。

未来提交可新增如下结构化信息，字段名仅作设计示意，当前严格schema尚不接受：

```json
{
  "primary_parent_id": "source-bound parent identifier",
  "required_feature_ids": ["typed feature evidence identifier"],
  "allowed_feature_substitutions": [],
  "selected_region_ids": ["registered actual subgraph identifier"],
  "allowed_attachment_port_ids": ["source-bound cut-site identifier"],
  "allowed_capping_port_ids": [],
  "feasibility_evidence_id": "measured connector-feasibility receipt",
  "selection_summary": "What is retained, what is removed, why the gain is plausible, and what remains uncertain."
}
```

程序核验这些ID与实际子图的一致性、关键原子/电荷/官能团是否存在以及几何可行性记录。Agent选择药物化学上的取舍，数值距离、坐标变换和接合可行性由程序负责。核心特征的重要性仍需配对实验验证，不能由LLM自报置信度直接转化为能量权重。

英文skills正文可按 **Baseline Preservation、Actual Subgraph Evidence、Complementary Feature Selection、Attachment Feasibility、Measured Feedback** 分模块。当前skills已提到完整基团、互补性和出口方向，下一版重点应是提供实际证据和可核验动作，而非增加泛化提醒或冗长推理文本。
