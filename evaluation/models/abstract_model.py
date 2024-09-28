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
Abstract model class to run evaluation. 
Should abstract away the prompt and loading of the model. 
Important: Cache all model answers.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class AbstractModel(ABC):
    registered_models = ["llama-8b", "baseline"]

    @abstractmethod
    def get_answers_and_cache(self, dataset) -> dict:
        """Should iterate over dataset and cache all answers.
        
        Returns: dict of answers (key: id in initial dataset, value: model_answer)"""
        pass

    @staticmethod
    def create(model_name, output_file_name, prompt_generator):
        import models.llama_8b
        import models.baseline
        registered_models = {
        "llama-8b": models.llama_8b.Llama8b,
        "baseline": models.baseline.Baseline
    }
        if model_name in registered_models:
            return registered_models[model_name](model_name=model_name, output_file_name=f"{output_file_name}_{model_name}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json", prompt_generator=prompt_generator)
        
        raise ValueError(f"Model {model_name} not found.")