
def set_seed(seed, name, quiet=False):
    import torch
    import numpy as np
    if not quiet:
        print(f"Setting {name} seed to {seed}")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)