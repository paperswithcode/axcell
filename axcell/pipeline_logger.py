#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re


class PipelineLogger:
    def __init__(self):
        self.observers = []

    def reset(self):
        self.observers = []

    def register(self, pattern, observer):
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.observers.append((pattern, observer))

    def unregister(self, pattern, observer):
        if pattern is None:
            self.observers = [(p, o) for p, o in self.observers if o != observer]
        else:
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            self.observers = [(p, o) for p, o in self.observers if o != observer or p.pattern != pattern.pattern]

    def __call__(self, step, **args):
        for pattern, observer in self.observers:
            if pattern.match(step):
                observer(step, **args)


pipeline_logger = PipelineLogger()
