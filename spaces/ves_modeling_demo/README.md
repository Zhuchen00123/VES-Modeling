# HF Space: VES Modeling（落地页 + 交互 Demo）

部署到 https://huggingface.co/spaces/235dsds/VES-Modeling

- `app.py`：Gradio 双标签页（About + verifier-first 交互 Demo）
- `requirements.txt`：gradio / pandas / numpy / scikit-learn / matplotlib
- 数据：Space 仓库内自带 `train.csv`（California Housing 公开训练集；Space 内部自行划分验证集，**不含隐藏标签**）

本地冒烟：`python -c "import app; print(app.evaluate('Random Forest', 0.000001))"`
