"""VES Modeling Space app — About landing + interactive verifier-first demo."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "train.csv")
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
}

ABOUT = """
## 🔬 VES Modeling — verifier-first executable search

> **AI can propose. It cannot grade itself.**

```
AI generates solution.py
        ↓
candidate claims RMSE = 0.000001
        ↓
VES ignores the claim
        ↓
Host recomputes RMSE = <真实值>
        ↓
Search improves candidate
```

**真实闭环已跑通**：LLM（deepseek-v4-flash）→ solution.py → Docker 沙箱 → predictions.json
→ 宿主 RegressionVerifier → Evidence → Judge(MINIMIZE rmse) → SearchEngine → improve
（2 drafts + 3 improves 全部 VERIFIED，rejected=0）。

📦 [GitHub](https://github.com/Zhuchen00123/VES-Modeling)
📊 [Dataset (California Housing)](https://huggingface.co/datasets/235dsds/VES-Modeling)
🧩 [VES Core](https://github.com/Zhuchen00123/Verified-Executable-Search)
"""


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(DATA_PATH)
    return frame.drop(columns=["target"]), frame["target"]


def evaluate(model_name: str, claimed_rmse: float) -> tuple[str, object]:
    """Host-side evaluation; returns markdown summary + scatter plot."""
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    host_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    host_mae = float(mean_absolute_error(y_val, preds))

    summary = (
        f"### 宿主验证结果（独立复算）\n\n"
        f"| 指标 | 候选自报 | 宿主复算 |\n"
        f"|---|---|---|\n"
        f"| RMSE | {claimed_rmse:.6f} | **{host_rmse:.6f}** |\n"
        f"| MAE | — | **{host_mae:.6f}** |\n\n"
        f"> 候选自报 RMSE 被**忽略**。排名只使用宿主复算值。\n\n"
        f"模型：**{model_name}** ｜ 验证样本：{len(y_val)}"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_val, preds, s=8, alpha=0.4)
    lim = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("actual (median house value, $100k)")
    ax.set_ylabel("predicted")
    ax.set_title(f"{model_name} — host RMSE {host_rmse:.4f}")
    fig.tight_layout()
    return summary, fig


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VES Modeling") as demo:
        gr.Markdown(
            "# 🔬 VES Modeling\n\n"
            "**Verifier-first executable search for computational modeling** ｜ "
            "🤗 喜欢请点右上角 Upvote"
        )
        with gr.Tabs():
            with gr.Tab("关于 / About"):
                gr.Markdown(ABOUT)
            with gr.Tab("交互 Demo"):
                gr.Markdown(
                    "选择模型并输入候选自报 RMSE，宿主在留出验证集上独立复算成绩。"
                )
                with gr.Row():
                    model = gr.Dropdown(
                        choices=list(MODELS), value="Linear Regression", label="候选模型"
                    )
                    claimed = gr.Number(
                        value=0.000001, label="候选自报 RMSE（会被忽略）"
                    )
                run = gr.Button("宿主验证", variant="primary")
                out = gr.Markdown()
                plot = gr.Plot(label="actual vs predicted")
                run.click(evaluate, inputs=[model, claimed], outputs=[out, plot])
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
