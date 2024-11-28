# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

from typing import Any, Callable, List

import numpy as np

from geom.projective_ops import pinhole2Kmat


def calib2intrinics_mapper(param: List[Any] = None) -> Callable[[str], np.ndarray]:
    """Create callable DUMMY function to return K matrix given the string input. """
    if len(param) == 4:
        def _return_callable(filename_for_query: str = '') -> np.ndarray:
            return pinhole2Kmat(tuple([float(elem) for elem in param]))

        return _return_callable
    elif param is None:
        return None
    else:
        raise NotImplementedError()
