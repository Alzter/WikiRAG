# ==================================================================================================================
# MoreHopQA Evaluation Benchmark
#
# This code was originally written by:
# Julian Schnitzler and Xanh Ho and Jiahao Huang and Florian Boudin and Saku Sugawara and Akiko Aizawa
#
# Retrieved from:
# https://github.com/Alab-NII/morehopqa/tree/27c72b1a220255093266a61f9e70af6ae981dc0b on 28/09/2024
#
# Licensed under the Apache License 2.0 (see LICENSE.TXT for details)

"""
Abstract dataset class to load a dataset and provide entries.
"""
from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    registered_datasets = ["morehopqa", "morehopqa-150"]
    
    @abstractmethod
    def items(self): 
        """Should iterate over data and return items"""
        pass

    @staticmethod
    def create(dataset_name):
        from datasets.morehopqa_loader import MorehopqaLoader, Morehopqa150Loader
        if dataset_name == "morehopqa":
            return MorehopqaLoader()
        elif dataset_name == "morehopqa-150":
            return Morehopqa150Loader()
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
        
