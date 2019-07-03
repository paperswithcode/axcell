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



def display_table(table, structure=None):
    """
        matrix - 2d ndarray with cell values
        strucutre - 2d ndarray with structure annotation
    """
    if hasattr(table, 'matrix'):
        matrix = table.matrix
    else:
        matrix = table
    if structure is None: structure = table.matrix_gold_tags
    html = []
    html.append(table_style)
    html.append('<div class="tableWrapper">')
    html.append("<table>")
    for row,struc_row in zip(matrix, structure):
        html.append("<tr>")
        for cell,struct in zip(row,struc_row):
            html.append(f'<td class="{struct}">{cell}</td>')
        html.append("</tr>")
    html.append("</table>")
    html.append('</div>')
    display_html("\n".join(html))
