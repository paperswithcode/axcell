#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

class PipelineError(Exception):
    pass


class UnpackError(PipelineError):
    pass


class LatexConversionError(PipelineError):
    pass
