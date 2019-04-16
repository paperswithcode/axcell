#!/usr/bin/env python

import fire
from sota_extractor.taskdb import TaskDB

def get_sota_tasks(filename):
    db = TaskDB()
    db.load_tasks(filename)
    return db.tasks_with_sota()


def label_tables(tasksfile):
    tasks = get_sota_tasks(tasksfile)
    for task in tasks:
        for dataset in task.datasets:
            for row in dataset.sota.rows:
                if 'arxiv.org' in row.paper_url:
                    for metric in row.metrics:
                        print((task.name, dataset.name, metric, row.model_name, row.metrics[metric], row.paper_url))
            

if __name__ == "__main__": fire.Fire(label_tables)
