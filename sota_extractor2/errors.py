class PipelineError(Exception):
    pass


class UnpackError(PipelineError):
    pass


class LatexConversionError(PipelineError):
    pass
