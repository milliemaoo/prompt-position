# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""

import numpy as np
import sklearn
import json
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import rf, Task
from lm_eval.utils import general_detokenize
from lm_eval.metrics import mean, acc_all, metric_max_over_ground_truths, yesno

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class CR(Task):
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    VERSION = 0
    TRAIN_PATH = "/mainfs/home/jm4n21/lm-evaluation-harness/dataset/CR/train.json"
    TEST_PATH = "/mainfs/home/jm4n21/lm-evaluation-harness/dataset/CR/test.json"

    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH) 
        self.max_length = 128

    def _load_json(self, file_path):
        with open(file_path) as file:
            return json.load(file)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._load_json(self.TEST_PATH)

    # def doc_to_text(self, doc):
    #     return "Question: Is this sentence positive or negative? Answer:\n{}\n".format(
    #         general_detokenize(doc["text"]),
    #     )

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = "Is this sentence positive or negative?\n"
        choice = "Options:\n- positive\n- negative\n"
        _input = general_detokenize(doc["text"])
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "
        pattern_mapping = {
            0: f"{prompt}{choice}{_input}{new_line}", 
            1: f"{prompt}{_input}{new_line}{choice}",
            2: f"{_input}{new_line}{prompt}{choice}" 
        }
        return f"{example_input_prefix}{pattern_mapping[pattern_id]}{example_output_prefix}"


    def doc_to_target(self, doc):
        return " {}".format({1: "positive", 0: "negative"}[doc["label"]])
    
    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, "positive") #如果是options格式应该改掉空格
        ll_negative, _ = rf.loglikelihood(ctx, "negative")
        return ll_positive, ll_negative

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
        else:
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return labeled_examples + example

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}