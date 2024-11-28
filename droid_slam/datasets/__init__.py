# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

import os
import importlib


class DynamicClassLoader:
    """ Provide an instance object that offers dynamic class loading depending on the parsed string. """

    def __init__(self, directory):
        self.directory = directory
        self._loaded_classes = {}

    def _load_class(self, class_name):
        """Load a first-time class."""
        if class_name not in self._loaded_classes:
            module = importlib.import_module(f'{self.directory}.{class_name}')
            klass = getattr(module, class_name)
            self._loaded_classes[class_name] = klass
        return self._loaded_classes[class_name]

    def __getattr__(self, class_name):
        return self._load_class(class_name)

    def get_class_by_name(self, class_name):
        """Load class implemented here from string."""
        return self._load_class(class_name)


directory_name = os.path.basename(os.path.dirname(__file__))  # Should be datasets

EvalDataSet = DynamicClassLoader(directory_name)  # Instance that provides dynamical class loading
