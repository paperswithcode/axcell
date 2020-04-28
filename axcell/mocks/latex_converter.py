#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

class LatexConverterMock:
    def __init__(self, mock_file):
        with open(mock_file, "r") as f:
            self.mock = f.read()

    def to_html(self, source_dir):
        return self.mock
