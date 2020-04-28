
#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

def set_seed(seed, name, quiet=False, all_gpus=True):
    import torch
    import numpy as np
    import random
    if not quiet:
        print(f"Setting {name} seed to {seed}")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if all_gpus:
        torch.cuda.manual_seed_all(seed)
