# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

from argparse import Namespace
from functools import partial
from typing import Any

from termcolor import cprint


def arg_has(args: Namespace, key: str, else_return: Any, none_str_is_none: bool = False) -> Any:
    """
    argparse handler to properly initialize.

    Parameters
    ----------
    args: Namespace
        Args to be fed into the downstream.
    key: str
        Keyname to be dealt with.
    else_return: Any
        What will be returned if the condition is not True.
    none_str_is_none: bool
        If the input value is NULL-string, override as None.
    Returns
    -------
    Any
        Updated value
    """
    ret = else_return if not key in args.__dict__.keys() else args.__dict__[key]
    if ret == 'None' and none_str_is_none:
        return None
    else:
        return ret


def args2dict(args_input: Namespace):
    """args to dict conversion."""
    return vars(args_input)


def args_override(args_input: Namespace, key: str, value: Any, override_warning=True) -> Namespace:
    """Update the args_input given its key and value."""
    args_curr = vars(args_input)
    prev_value = args_curr[key]
    args_curr.update({key: value})
    if override_warning:
        cprint('## [DEBUG] Arg=`{}` is overriden: {} -> {}'.format(key, prev_value, args_curr[key]), 'yellow')
    ret_args = Namespace(**args_curr)
    return ret_args


arg_has_false = partial(arg_has, else_return=False)
