#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from .. import config # to get logging init

logger = logging.getLogger(__name__)

try:
    from db import *
except:
    logger.info("Unable to intialise django falling back to json data")
    from json import *
