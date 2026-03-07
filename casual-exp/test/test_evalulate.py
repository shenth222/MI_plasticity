"""
加载本地模型与本地 GLUE 数据集，在各子任务上运行评估并打印结果。

使用方式:
  python test/test_evalulate.py
"""

import os
import sys
from typing import Dict, Any, List, Optional

# 项目根为 casual-exp 时可直接 import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluate import GLUE_TASK_CONFIGS, evaluate_glue


# ---------- 配置：本地模型与 GLUE 数据路径 ----------
LOCAL_MODEL_PATH = "/data1/shenth/models/deberta/v3-base"   # 本地模型/checkpoint 根目录或单一路径，例如: "/data1/shenth/models/deberta/v3-base"
LOCAL_GLUE_PATH = "/data1/shenth/datasets/glue"   # 本地 GLUE 数据根目录，例如: "/data1/shenth/datasets/glue"

# 要评估的子任务（默认全部）；若某任务无对应模型会跳过
TASKS: List[str] = list(GLUE_TASK_CONFIGS.keys())   # cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli

# 评估超参
BATCH_SIZE = 16
MAX_LENGTH = 256


def _model_path_for_task(base_path: str, task: str) -> Optional[str]:
    """每个任务优先使用 base_path/task 子目录，不存在则用 base_path。"""
    if not base_path or not os.path.exists(base_path):
        return None
    per_task = os.path.join(base_path, task)
    if os.path.exists(per_task):
        return per_task
    return base_path


def run_all_glue_evaluations(
    model_path: str = "",
    dataset_path: str = "",
    tasks: Optional[List[str]] = None,
    batch_size: int = 16,
    max_length: int = 256,
) -> Dict[str, Any]:
    """
    在指定 GLUE 子任务上依次评估，返回 { task_name: metrics }，失败任务为 { task_name: {"error": str} }。
    """
    model_path = model_path or LOCAL_MODEL_PATH
    dataset_path = dataset_path or LOCAL_GLUE_PATH
    tasks = tasks or TASKS

    if not model_path or not os.path.exists(model_path):
        return {"_error": f"模型路径不存在或未配置: {model_path}"}
    if not dataset_path or not os.path.exists(dataset_path):
        return {"_error": f"GLUE 数据路径不存在或未配置: {dataset_path}"}

    results: Dict[str, Any] = {}
    for task in tasks:
        if task not in GLUE_TASK_CONFIGS:
            results[task] = {"error": f"未知任务: {task}"}
            continue
        path = _model_path_for_task(model_path, task)
        if path is None:
            results[task] = {"error": "无可用模型路径"}
            continue
        try:
            metrics = evaluate_glue(
                path,
                task_name=task,
                dataset_path=dataset_path,
                batch_size=batch_size,
                max_length=max_length,
            )
            results[task] = metrics
        except Exception as e:
            results[task] = {"error": str(e)}
    return results


def _main_metric(task: str, metrics: Dict[str, Any]) -> str:
    """取该任务的主指标用于表格展示。"""
    if "error" in metrics:
        return f"ERR: {metrics['error'][:40]}..."
    if task == "mnli" and "accuracy_matched" in metrics:
        m = metrics.get("accuracy_matched", 0)
        mm = metrics.get("accuracy_mismatched", 0)
        return f"matched={m:.4f}  mismatched={mm:.4f}"
    cfg = GLUE_TASK_CONFIGS.get(task, {})
    names = cfg.get("metric_names", ["accuracy"])
    parts = []
    for k in names:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
    return " ".join(parts) if parts else str(metrics)


def print_results(results: Dict[str, Any]) -> None:
    """打印各子任务结果：先汇总表，再分任务详情。"""
    if not results:
        print("无结果可打印。")
        return
    if "_error" in results:
        print("运行失败:", results["_error"])
        return

    # 汇总表
    print("\n" + "=" * 60)
    print("GLUE 各子任务评估结果汇总")
    print("=" * 60)
    print(f"{'任务':<10} {'主指标':<50}")
    print("-" * 60)
    for task in TASKS:
        if task not in results:
            continue
        m = _main_metric(task, results[task])
        print(f"{task:<10} {m:<50}")
    print("=" * 60)

    # 分任务详情
    print("\n--- 各子任务详细指标 ---\n")
    for task in TASKS:
        if task not in results:
            continue
        metrics = results[task]
        if "error" in metrics:
            print(f"[{task}] 失败: {metrics['error']}\n")
            continue
        print(f"[{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()


def main() -> Dict[str, Any]:
    """加载本地模型与 GLUE 数据，跑全子任务评估并打印。"""
    print("配置:")
    print(f"  本地模型路径: {LOCAL_MODEL_PATH or '(未设置)'}")
    print(f"  本地 GLUE 路径: {LOCAL_GLUE_PATH or '(未设置)'}")
    print(f"  待评估任务: {TASKS}")
    print(f"  batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}")

    results = run_all_glue_evaluations(
        model_path=LOCAL_MODEL_PATH,
        dataset_path=LOCAL_GLUE_PATH,
        tasks=TASKS,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    print_results(results)
    return results


if __name__ == "__main__":
    main()

