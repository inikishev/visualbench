import shutil
import os
from typing import TYPE_CHECKING

from ..utils.python_tools import format_number
from .run import Task

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _maybe_format(x):
    if isinstance(x, float): return format_number(x, 3)
    return x

def _dict_to_str(d: dict):
    return ' '.join([f"{k}={_maybe_format(v)}" for k,v in d.items()])

def print_task_summary(root: str, task_name:str, metric: str = "train loss", maximize=False) -> None:
    task = Task.load(os.path.join(root, task_name), load_loggers=False, decoder=None)
    sweeps = task.best_sweeps(metric, maximize, n=1000)
    runs = [s.best_runs(metric, maximize, n=1)[0] for s in sweeps]

    for i, r in enumerate(runs):
        key = 'max' if maximize else 'min'
        if len(r.hyperparams) == 0: n = f"{i+1}: {r.run_name}"
        else: n = f"{i+1}: {r.run_name} ({_dict_to_str(r.hyperparams)})"
        print(n.ljust(100)[:100], f"{format_number(r.stats[metric][key], 5)}")


def rename_run(root:str, old: str, new:str) -> None:
    renamed = False
    for task in os.listdir(root):
        task_path = os.path.join(root, task)
        for run in os.listdir(task_path):
            if run == old:
                renamed = True
                run_path = os.path.join(task_path, run)
                os.rename(run_path, os.path.join(task_path, new))

    summaries_root = f'{root} - summaries'
    if os.path.exists(summaries_root):
        for run in os.listdir(summaries_root):
            if run == old:
                renamed = True
                run_path = os.path.join(summaries_root, run)
                os.rename(run_path, os.path.join(summaries_root, new))

    if not renamed:
        raise FileNotFoundError(f"{old} doesn't exist")


def delete_run(root:str, name:str) -> None:
    deleted = False
    for task in os.listdir(root):
        task_path = os.path.join(root, task)
        for run in os.listdir(task_path):
            if run == name:
                deleted = True
                shutil.rmtree(os.path.join(task_path, run))

    summaries_root = f'{root} - summaries'
    if os.path.exists(summaries_root):
        for run in os.listdir(summaries_root):
            if run == name:
                deleted = True
                shutil.rmtree(os.path.join(summaries_root, run))

    if not deleted:
        raise FileNotFoundError(f"{name} doesn't exist")

def rename_task(root:str, old: str, new:str) -> None:
    renamed = False
    for task in os.listdir(root):
        if task == old:
            renamed = True
            task_path = os.path.join(root, task)
            os.rename(task_path, os.path.join(root, new))

    summaries_root = f'{root} - summaries'
    if os.path.exists(summaries_root):
        for run in os.listdir(summaries_root):
            run_path = os.path.join(summaries_root, run)
            for task in os.listdir(run_path):
                if task.replace('.png', '') == old:
                    renamed = True
                    task_path = os.path.join(run_path, task)
                    os.rename(task_path, os.path.join(run_path, f'{new}.png'))


    if not renamed:
        raise FileNotFoundError(f"{old} doesn't exist")
