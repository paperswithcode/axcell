#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unidecode import unidecode
import re

# cleaning & normalization
parens_re = re.compile(r"\([^)]*?\)|\[[^]]*?\]")

strip_nonalnum_re = re.compile(r"^\W*(\w.*\b)\W*$")
def strip_nonalnum(s):
    m = strip_nonalnum_re.match(s)
    if m:
        return m.group(1)
    return ""

def remove_parens(text):
    return parens_re.sub("", text)

def clean_name(name):
    return remove_parens(unidecode(name).strip()).strip()

def clean_cell(cell):
    return strip_nonalnum(clean_name(cell))

year_2k_re = re.compile(r"20(\d\d)")
hyphens_re = re.compile(r"[-_'`–’→]")
ws_re = re.compile(r"\s+")


refs_re = re.compile(r"(xxtable-)?xxanchor-[^ ]*|xxref-[^ ]*")

def remove_references(s):
    return refs_re.sub("", s)

def normalize_dataset_ws(name):
    name = remove_references(name)
    name = hyphens_re.sub(" ", name)
    name = year_2k_re.sub(r"\1", name)
    name = ws_re.sub(" ", name)
    return unidecode(name.strip().lower())

def normalize_dataset(name):
    name = remove_references(name)
    name = year_2k_re.sub(r"\1", name)
    name = hyphens_re.sub("", name)
    name = ws_re.sub(" ", name)
    return unidecode(name.strip().lower())


def normalize_cell(s):
    return unidecode("".join([x for x in s if x.isalnum()]))

def normalize_cell_ws(s):
    return unidecode("".join([x for x in s if x.isalnum() or x.isspace()]))

# end of cleaning & normalization
