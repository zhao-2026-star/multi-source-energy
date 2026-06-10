# -*- coding: utf-8 -*-
"""运行管理：自动分配 run_id，统一目录结构。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = os.path.join(BASE_DIR, "runs")


def get_next_run_id():
    """自动检测下一个 run 编号 (run1, run2, ...)。"""
    os.makedirs(RUN_BASE, exist_ok=True)
    existing = [d for d in os.listdir(RUN_BASE)
                if d.startswith("run") and d[3:].isdigit()]
    if not existing:
        return 1
    return max(int(d[3:]) for d in existing) + 1


def get_run_dir(run_id):
    return os.path.join(RUN_BASE, f"run{run_id}")


def get_weight_dir(run_id):
    return os.path.join(get_run_dir(run_id), "weight")


def get_figure_dir(run_id):
    return os.path.join(get_run_dir(run_id), "figure")


def get_log_dir(run_id):
    return os.path.join(get_run_dir(run_id), "log")


def get_prediction_dir(run_id):
    return os.path.join(get_run_dir(run_id), "prediction")


def get_result_dir(run_id):
    return os.path.join(get_run_dir(run_id), "result")
