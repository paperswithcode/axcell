#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pytest
from decimal import Decimal
from axcell.models.linking.bm25_naive import convert_metric

raw_values = ["0.21", "0.21%", "21", "21%"]
ranges = ["0-1", "1-100", "abs", ""]

values = {
  #             0.21      0.21%      21      21%
  "0-1":   [   "0.21", "0.0021",  "0.21", "0.21"],
  "1-100": [     "21",   "0.21",    "21",   "21"],
  "abs":   [   "0.21", "0.0021",    "21", "0.21"],
  "":      [   "0.21",   "0.21",    "21",   "21"]
}

comp_values = {
   #              0.21      0.21%      21      21%
   "0-1":    [   "0.79", "0.9979",  "0.79", "0.79"],
   "1-100":  [     "79",  "99.79",    "79",   "79"],
   "":       [   "0.79",  "99.79",    "79",   "79"]
}

cases = [(raw_value, rng, complementary, Decimal(answer))
    for complementary, vals in zip([False, True], [values, comp_values])
    for rng in vals
    for raw_value, answer in zip(raw_values, vals[rng])
]


@pytest.mark.parametrize("raw_value,rng,complementary,expected", cases)
def test_ranges(raw_value, rng, complementary, expected):
    value = convert_metric(raw_value, rng, complementary)
    assert value == expected, (f"{'complement of ' if complementary else ''}"
        f"raw value {raw_value}, assuming {rng if rng else 'empty'} range "
        f"should be extracted as {expected}, not {value}")
