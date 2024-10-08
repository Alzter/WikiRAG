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
Load dataset based on 2wikihop dataset.
"""
from datasets.abstract_dataset_loader import DatasetLoader
import json
import random
random.seed(42)

class MorehopqaLoader(DatasetLoader):
    path = "datasets/files/morehopqa_final.json"

    def __init__(self):
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.length = len(self.data)

    def items(self):
        for item in self.data:
            yield item


class Morehopqa150Loader(DatasetLoader):
    path = "datasets/files/morehopqa_final_150samples.json"

    def __init__(self):
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.length = len(self.data)

    def items(self):
        for item in self.data:
            yield item