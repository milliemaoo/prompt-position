"""
GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
https://openreview.net/pdf?id=rJ4km2R5t7

The General Language Understanding Evaluation (GLUE) benchmark is a collection of
resources for training, evaluating, and analyzing natural language understanding
systems. GLUE consists of:
- A benchmark of nine sentence- or sentence-pair language understanding tasks built
on established existing datasets and selected to cover a diverse range of dataset
sizes, text genres, and degrees of difficulty, and
- A diagnostic dataset designed to evaluate and analyze model performance with
respect to a wide range of linguistic phenomena found in natural language.

Homepage: https://gluebenchmark.com/
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize
import json


# TODO(jon-tow): Add citations for the individual datasets/tasks that make up GLUE.
_CITATION = """
@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355",
    abstract = "Human ability to understand language is \textit{general, flexible, and robust}. In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data. If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a unified model that can execute a range of linguistic tasks across different domains. To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE, gluebenchmark.com): a benchmark of nine diverse NLU tasks, an auxiliary dataset for probing models for understanding of specific linguistic phenomena, and an online platform for evaluating and comparing models. For some benchmark tasks, training data is plentiful, but for others it is limited or does not match the genre of the test set. GLUE thus favors models that can represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks. While none of the datasets in GLUE were created from scratch for the benchmark, four of them feature privately-held test data, which is used to ensure that the benchmark is used fairly. We evaluate baselines that use ELMo (Peters et al., 2018), a powerful transfer learning technique, as well as state-of-the-art sentence representation models. The best models still achieve fairly low absolute scores. Analysis with our diagnostic dataset yields similarly weak performance over all phenomena tested, with some exceptions.",
}
"""


# Single-Sentence Tasks


class CoLA(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "cola"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: Does this sentence make sense?\nAnswer:".format(
            doc["sentence"]
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["sentence"]

    def doc_to_target(self, doc):
        return " {}".format({1: "yes", 0: "no"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " yes")
        ll_false, _ = rf.loglikelihood(ctx, " no")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        gold = doc["label"]
        return {"mcc": (gold, pred)}

    def higher_is_better(self):
        return {"mcc": True}

    def aggregation(self):
        return {"mcc": matthews_corrcoef}


class SST(Task):
    VERSION = 0
    TRAIN_PATH = "/mainfs/scratch/jm4n21/lm-evaluation-harness/dataset/sst-2/train_1.json"
    TEST_PATH = "/mainfs/scratch/jm4n21/lm-evaluation-harness/dataset/sst-2/test.json"

    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH) 
        self.max_length = 128

    def _load_json(self, file_path):
        with open(file_path) as file:
            return json.load(file)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self._load_json(self.TEST_PATH)
        
    def doc_to_text(self, doc, pattern_id):
        prompt_templates = [
                "Question: Is this sentence positive or negative?\nAnswer:\n{input}",
                "Question: Is this sentence positive or negative?\n{input}\nAnswer:",
                "{input}\nQuestion: Is this sentence positive or negative?\nAnswer:"
            ]
        prompt = prompt_templates[pattern_id].format(input=general_detokenize(doc["sentence"]))

        return prompt

    def doc_to_target(self, doc):
        return " {}".format({1: "positive", 0: "negative"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " positive")
        ll_negative, _ = rf.loglikelihood(ctx, " negative")
        return ll_positive, ll_negative

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
        else:
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, pattern_id) + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, pattern_id)
        return labeled_examples + example

    def process_results(self, doc, results, pattern_id):
        num_classes = 2
        ll_positive, ll_negative = results

        #train_5
        # if pattern_id == 0:
        #     p_cf = np.array([0.68890432,0.31109568])
        # elif pattern_id == 1:
        #     p_cf = np.array([0.47323628,0.52676372])
        # elif pattern_id == 2:
        #     p_cf = np.array([0.43976171,0.56023829])

        #train_4
        # if pattern_id == 0:
        #     p_cf = np.array([0.62260784,0.37739216])
        # elif pattern_id == 1:
        #     p_cf = np.array([0.65790677,0.34209323])
        # elif pattern_id == 2:
        #     p_cf = np.array([0.42521825,0.57478175])

        #train_3:
        # if pattern_id == 0:
        #     p_cf = np.array([0.60642719,0.39357281])
        # elif pattern_id == 1:
        #     p_cf = np.array([0.30067287,0.69932713])
        # elif pattern_id == 2:
        #     p_cf = np.array([0.82260674,0.17739326])

        #train_2
        # if pattern_id == 0:
        #     p_cf = np.array([0.82658253,0.17341747])
        # elif pattern_id == 1:
        #     p_cf = np.array([0.8590389,0.1409611])
        # elif pattern_id == 2:
        #     p_cf = np.array([0.72854186,0.27145814])

        #train_1
        if pattern_id == 0:
            p_cf = np.array([0.72896575,0.27103425])
        elif pattern_id == 1:
            p_cf = np.array([0.83104179,0.16895821])
        elif pattern_id == 2:
            p_cf = np.array([0.7035439,0.2964561])

        W = np.linalg.inv(np.identity(num_classes) * p_cf)
        b = np.zeros([num_classes, 1])

        probs = np.exp(np.array([ll_negative, ll_positive]))
        probs = probs / np.sum(probs)  # 标准化概率分布        
        calibrated_probs = np.matmul(W, np.expand_dims(probs, axis=-1)) + b

        pred = np.argmax(calibrated_probs)

        # pred = ll_positive > ll_negative
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


# Inference Tasks


class MNLI(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test_matched"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["premise"],
            doc["hypothesis"].strip()
            + ("" if doc["hypothesis"].strip().endswith(".") else "."),
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class MNLIMismatched(MNLI):
    VERSION = 0

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation_mismatched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test_mismatched"]


class QNLI(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return (
            "{}\n{}\nQuestion: Does this response answer the question?\nAnswer:".format(
                doc["question"],
                doc["sentence"],
            )
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = not entailment
        return " {}".format({0: "yes", 1: "no"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        pred = ll_no > ll_yes
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class WNLI(Task):
    VERSION = 1
    DATASET_PATH = "glue"
    DATASET_NAME = "wnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True or False?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = not_entailment
        return " {}".format({0: "False", 1: "True"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class RTE(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "rte"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True or False?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # 0 = entailment
        # 1 = not_entailment
        return " {}".format({0: "True", 1: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


# Similarity and Paraphrase Tasks


class MRPC(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mrpc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing?\nAnswer:".format(
            general_detokenize(doc["sentence1"]),
            general_detokenize(doc["sentence2"]),
        )

    def doc_to_target(self, doc):
        return " {}".format(yesno(doc["label"]))

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class QQP(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qqp"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "Question 1: {}\nQuestion 2: {}\nQuestion: Do both questions ask the same thing?\nAnswer:".format(
            doc["question1"],
            doc["question2"],
        )

    def doc_to_target(self, doc):
        return " {}".format(yesno(doc["label"]))

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class STSB(Task):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "stsb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "sentence 1: {}\nsentence 2: {}\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["label"])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")
