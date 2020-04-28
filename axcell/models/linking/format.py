#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, InvalidOperation

float_value_re = re.compile(r"([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
float_value_nc = re.compile(r"(?:[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
par_re = re.compile(r"\{([^\}]*)\}")
escaped_whitespace_re = re.compile(r"(\\\s)+")

def format_to_regexp(format):
    placeholders = par_re.split(format.strip())
    regexp = ""
    fn=lambda x: x
    for i, s in enumerate(placeholders):
        if i % 2 == 0:
            if s.strip() == "":
                regexp += escaped_whitespace_re.sub(r"\\s+", re.escape(s))
            else:
                regexp += escaped_whitespace_re.sub(r"\\s*", re.escape(s))
        elif s.strip() == "":
            regexp += float_value_nc.pattern
        else:
            regexp += float_value_re.pattern
            ss = s.strip()
            if ss == "100*x" or ss == "100x":
                fn = lambda x: 100*x
            elif ss == "x/100":
                fn = lambda x: x/100
    #return re.compile('^'+regexp+'$'), fn
    return re.compile('^' + regexp), fn

def extract_value(cell_value, format):
    cell_value = re.sub(r"\s+%", "%", cell_value).replace(",", "")
    cell_value = cell_value.replace("(", " ").replace(")", " ").strip()
    regexp, fn = format_to_regexp(format)
    match = regexp.match(cell_value)
    if match is None or not len(match.groups()):
        return Decimal('NaN')
    return fn(Decimal(match.group(1)))
