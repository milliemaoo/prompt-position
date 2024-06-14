# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize



# TODO: Replace `NewTask` with the name of your Task.
class TREC(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "trec"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return False

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return "Answer: Categories: Abbreviation, Entity, Description, Person, Location, Quantity\n\nWhat category best describes: {}\n".format(
                    doc["text"],
                )
        # return "Categories: Abbreviation, Entity, Description, Person, Location, Quantity\n\nWhat category best describes: {}\nAnswer:".format(
        #             doc["text"],
        #         )

    def doc_to_target(self, doc):
        return " {}".format({0: "Abbreviation", 1: "Entity", 2: "Description", 3: "Person", 4: "Location", 5: "Quantity"}[doc["coarse_label"]])

    def construct_requests(self, doc, ctx):
        ll_Abbreviation, _ = rf.loglikelihood(ctx, " Abbreviation")
        ll_Entity, _ = rf.loglikelihood(ctx, " Entity")
        ll_Description, _ = rf.loglikelihood(ctx, " Description")
        ll_Person, _ = rf.loglikelihood(ctx, " Person")
        ll_Location, _ = rf.loglikelihood(ctx, " Location")
        ll_Quantity, _ = rf.loglikelihood(ctx, " Quantity")

        return ll_Abbreviation, ll_Entity, ll_Description, ll_Person, ll_Location, ll_Quantity

    def process_results(self, doc, results):
        gold = doc["coarse_label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}