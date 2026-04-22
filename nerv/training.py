import os
import re
from typing import Optional


def find_old_slurm_id(ckpt_dir: str) -> Optional[str]:
    parent = os.path.dirname(os.path.dirname(ckpt_dir))
    if not os.path.isdir(parent):
        return None
    for name in os.listdir(parent):
        match = re.search(r"-(\d+)$", name)
        if match:
            return match.group(1)
    return None
