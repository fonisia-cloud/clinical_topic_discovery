# Clinical Topic Discovery 使用说明（中文）

本指南面向要把应用分享给同事/朋友的场景，覆盖安装、依赖、功能说明、使用流程和常见问题。

## 1. 软件定位

Clinical Topic Discovery 是一个轻量本地应用，用于临床论文前期选题：

- 快速检索 PubMed 文献
- 自动结构化整理标题/摘要/期刊/研究类型
- 输出证据地图与趋势分析
- 给出候选选题及支撑 PMID

当前版本聚焦“选题发现”，不包含正式论文全自动写作。

## 2. 运行环境要求

- 操作系统：Windows 10/11（macOS/Linux 也可运行）
- Python：3.10+（推荐 3.11/3.12）
- 网络：可访问 `https://eutils.ncbi.nlm.nih.gov`

## 3. 依赖说明（轻量）

核心依赖见 `requirements.txt`：

- `streamlit`：Web UI
- `requests`：PubMed API 调用
- `pandas`：数据处理
- `matplotlib`：基础图表

数据库使用 Python 内置 `sqlite3`，不需要单独安装 MySQL/PostgreSQL。

## 4. 安装步骤

在项目目录 `clinical_topic_discovery` 下执行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

启动应用：

```bash
streamlit run app.py
```

默认访问：`http://localhost:8501`

## 5. 给朋友分享的两种方式

### 方式 A：分享源码（最简单）

把整个 `clinical_topic_discovery` 文件夹打包给对方，让对方按“安装步骤”执行。

### 方式 B：你本机启动，对方局域网访问

1. 你启动：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

2. 告诉对方访问：`http://你的局域网IP:8501`
3. 若访问失败，检查防火墙是否放行 8501 端口。

> 注意：默认是本地服务，不是公网部署。若需公网可再加反向代理/云服务器。

## 6. 功能介绍

### 6.1 Search Settings（左侧）

- `Project name`：本次检索任务名称
- `Keywords`：支持布尔语法 `AND/OR/NOT`；逗号仍按 `OR` 处理（默认留空）
- `MeSH terms`：支持布尔语法 `AND/OR/NOT`；逗号仍按 `OR` 处理（默认留空）
- `Keyword matching mode`：
  - `PubMed default`：更接近 PubMed 网页检索框（自动术语映射）
  - `Title/Abstract only`：仅在题名与摘要字段匹配，结果更严格
- `Start year / End year`：时间范围
- `Study types`：研究类型过滤（RCT、Meta-analysis 等）
- `Max papers`：最多抓取文献数量
- `NCBI email / api_key`：可选，提升调用稳定性和配额

点击 `Run Search and Analyze` 执行检索。

### 6.2 Search Results

- 表格展示文献结构化字段
- 支持按 `Publication year`、`Citation count`、`JCR IF`、`CAS major tier`、`Journal impact proxy` 排序
- 点击 `Fetch citation metrics` 可补充引用数（OpenAlex）
- 点击 `Fetch JCR IF` 可整合 ShowJCR 的 JCR2022/2023/2024 数据
- 点击 `Fetch CAS partition` 可整合 ShowJCR 的中科院分区表（FQBJCR2025/2023/2022）
- 支持摘要预览长度调节
- `Abstract viewer` 可查看单篇完整摘要
- 可导出当前筛选结果 CSV

### 6.3 Evidence Map

- 疾病/MeSH 信号
- 干预、结局、研究设计分布
- 用于快速识别证据聚集方向

### 6.4 Trend Insights

- 年度发文趋势
- 高频术语
- 新兴术语（近期/历史增长比）

### 6.5 Topic Ideas

- 一键生成“最终选题版本”：规则引擎 + 可选LLM增强（非双轨展示）
- 输出每个选题的理由、证据缺口、支持 PMID
- 可下载 Topic Evidence JSONL 供外部模型复核
- 可调三类权重：
  - `Clinical value`
  - `Innovation`
  - `Feasibility`
- 权重用于“当前 run 的重排序”，不影响检索逻辑
- 点 `Save current ranking to this run` 才会保存当前排序

## 7. 标准使用流程（建议）

1. 先用宽一点的关键词做第一轮检索（100~300 篇）
2. 在 Search Results 看摘要，剔除明显无关方向
3. 在 Evidence Map 找到高密度人群/干预/结局组合
4. 在 Trend Insights 看近年增长和新兴词
5. 在 Topic Ideas 调权重对比两种策略：
   - 创新优先（Innovation 高）
   - 可行优先（Feasibility 高）
6. 导出 CSV + 记录候选题及 PMID，进入后续写作阶段

## 8. 数据与文件位置

- SQLite 数据库：`data/app.db`
- ShowJCR 数据库（自动下载）：`data/showjcr_jcr.db`
- 导出 CSV：浏览器下载目录

## 9. 常见问题（FAQ）

### Q1：为什么和 PubMed 网页数量不一致

常见原因：

- 网页端默认行为接近自动术语映射；应用若设为 `Title/Abstract only` 会更严格
- 应用会显示 `PubMed matched X`（总命中）与 `retrieved top Y`（实际抓取上限，受 `Max papers` 影响）
- 研究类型过滤、时间范围、MeSH 限定若不完全一致，数量会不同

### Q2：检索后没有结果

- 放宽关键词（减少限定词）
- 放宽年份范围
- 减少研究类型过滤条件
- 检查网络是否可访问 NCBI

### Q3：摘要为空

部分 PubMed 记录本身无摘要或摘要受限，属正常现象。

### Q4：权重改了但排序变化小

当候选题基础证据结构很相近时，变化会较小。可尝试：

- 扩大 `Max papers`
- 放宽检索词以增加候选多样性
- 将某一项权重显著提高（如 80）做极端对比

### Q5：端口占用无法启动

换端口启动：

```bash
streamlit run app.py --server.port 8502
```

## 10. 安全与合规提示

- 本工具用于科研选题支持，不用于临床诊疗决策
- JCR IF 与中科院分区来自 ShowJCR 数据库，引用数来自 OpenAlex；使用前请核对版本年份与口径
- 结果应由研究者二次核验，尤其是摘要级结论
- 对外分享时请注意本地数据库中是否有敏感备注

## 11. 版本说明（当前）

- 版本：V1
- 已实现：检索、证据图谱、趋势、手动纳入集、规则引擎+LLM一体化选题
- 后续建议：句级证据映射、Word导出、审稿规则检查
