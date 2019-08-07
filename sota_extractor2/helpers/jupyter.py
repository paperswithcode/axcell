from IPython.core.display import display, HTML
from .table_style import table_style
def set_seed(seed, name):
    import torch
    import numpy as np
    print(f"Setting {name} seed to {seed}")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def display_html(s): return display(HTML(s))



def display_table(table, structure=None, layout=None):
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
    html = []
    html.append(table_style)
    html.append('<div class="tableWrapper">')
    html.append("<table>")
    for row,struc_row, layout_row in zip(matrix, structure, layout):
        html.append("<tr>")
        for cell,struct,layout in zip(row,struc_row,layout_row):
            html.append(f'<td class="{struct} {layout}">{cell}</td>')
        html.append("</tr>")
    html.append("</table>")
    html.append('</div>')
    display_html("\n".join(html))
