#!/usr/bin/env python3
#
# Created on Mon Nov 11 2024 17:10:17
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import builtins
import os
import os.path as osp

import torch.multiprocessing as mp
from rich import print as rich_print
from rich.pretty import install

CURRENT_DIR = osp.dirname(osp.realpath(__file__))
BASE_DIR = osp.realpath(osp.join(CURRENT_DIR, ".."))
os.environ["USF_PROJECT_DIRECTORY"] = BASE_DIR

# Load .env into os.environ so Hydra's ${oc.env:VAR} resolvers pick up
# user-local settings.  Does not overwrite already-set env vars.
from dotenv import load_dotenv

load_dotenv(osp.join(BASE_DIR, ".env"), override=False)


# Hook pprint to use Rich
install()


def scrub_state_dict(obj: object) -> object:
    """Recursively replace ``state_dict`` and ``optimizer_states`` values with placeholder strings.

    Prevents Rich from pretty-printing massive tensors when a checkpoint dict
    is accidentally passed to ``print()``.

    Args:
        obj (object): Any Python object (typically a nested dict from a checkpoint).

    Returns:
        object: A copy with large state entries replaced by summary strings.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ["state_dict", "optimizer_states"]:
                out[k] = f"{k} object"
            else:
                out[k] = scrub_state_dict(v)
        return out
    if isinstance(obj, list):
        return [scrub_state_dict(e) for e in obj]
    return obj


def safe_print(*args, **kwargs) -> None:
    """Rich-pretty-print with state-dict scrubbing to avoid dumping large tensors.

    Drop-in replacement for ``builtins.print`` monkey-patched at import time.
    """
    out_args = [scrub_state_dict(arg) for arg in args]
    out_kwargs = {k: scrub_state_dict(v) for k, v in kwargs.items()}
    return rich_print(*out_args, **out_kwargs)


# Monkey-patch the built-in print to be safe_print
builtins.print = safe_print

mp.set_start_method("fork", force=True)
mp.set_sharing_strategy("file_system")
