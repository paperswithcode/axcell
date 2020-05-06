#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from IPython.core.display import display, HTML
from .table_style import table_style
import numpy as np


def set_seed(seed, name):
    import torch
    import numpy as np
    print(f"Setting {name} seed to {seed}")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def display_html(s): return display(HTML(s))


def table_to_html(table, structure=None, layout=None, predictions=None, tooltips=None):
    """
        matrix - 2d ndarray with cell values
        strucutre - 2d ndarray with structure annotation
    """
    if hasattr(table, 'matrix'):
        matrix = table.matrix
    else:
        matrix = table
    if structure is None: structure = table.matrix_gold_tags
    if layout is None: layout = np.zeros_like(matrix, dtype=str)
    if predictions is None: predictions = np.zeros_like(matrix, dtype=str)
    if tooltips is None: tooltips = np.zeros_like(matrix, dtype=str)
    html = []
    html.append(table_style)
    html.append('<div class="tableWrapper">')
    html.append("<table>")
    for row,struc_row, layout_row, preds_row, tt_row in zip(matrix, structure, layout, predictions, tooltips):
        html.append("<tr>")
        for cell,struct,layout,preds, tt in zip(row,struc_row,layout_row,preds_row, tt_row):
            html.append(f'<td class="{struct} {layout} {preds}" title="{tt}">{cell}</td>')
        html.append("</tr>")
    html.append("</table>")
    html.append('</div>')
    return "\n".join(html)


def display_table(table, structure=None, layout=None):
    html = table_to_html(table, structure, layout)
    display_html(html)
