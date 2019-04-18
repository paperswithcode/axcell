#!/usr/bin/env python

import fire
from label_tables import get_sota_tasks


def normalize_metrics(tasksfile):
    tasks = get_sota_tasks(tasksfile)

    print("task\tdataset\tmetric")
    for task in tasks:
        for dataset in task.datasets:
            for row in dataset.sota.rows:
                for metric in row.metrics:
                    print(f"{task.name}\t{dataset.name}\t{metric}")


if __name__ == "__main__": fire.Fire(normalize_metrics)
