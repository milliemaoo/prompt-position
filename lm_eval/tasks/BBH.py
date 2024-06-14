import os
import json
import hashlib
import functools
import numpy as np
import re
import importlib.resources
import evaluate
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_DEFAULT_REGEX = r"[^\.\?\!\;\n]+"
exact_match = evaluate.load("/.../lm-evaluation-harness/lm_eval/exact_match.py")

class causal_judgement(Task):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/causal_judgement/train.json"
    TEST_PATH = "/.../lm-evaluation-harness/dataset/causal_judgement/test.json"

    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH) 
        self._description = "Answer questions about causal attribution.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = "How would a typical person answer each of the following questions about causation?\n"
        choice = "Options:\n- Yes\n- No\n"
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "

        if pattern_id == 0:
            return f"{example_input_prefix}{prompt}{choice}{doc['input']}{new_line}{example_output_prefix}"
        elif pattern_id == 1:
            return f"{example_input_prefix}{prompt}{doc['input']}{new_line}{choice}{example_output_prefix}"
        elif pattern_id == 2:
            return f"{example_input_prefix}{doc['input']}{new_line}{choice}{prompt}{example_output_prefix}"

    def doc_to_target(self, doc):
        return max(doc["target_scores"].items(), key=lambda x: x[1])[0]

    def _doc_to_queries(self, doc):
        return list(doc["target_scores"].keys())

    def construct_requests(self, doc, ctx):
        requests = []
        if self._has_multi_choice:
            queries = self._doc_to_queries(doc)
            requests += [
                rf.loglikelihood(ctx, continuation)[0] for continuation in queries
            ]
        if self._has_generative:
            requests.append(
                rf.greedy_until(ctx, {"until": ["\nQ:","\n\nQ:","\n\n"," Q:"], "max_length": self.max_length})
            )
        return requests

    def process_results(self, doc, results):
        res = {}
        for metric in self.metrics:
            if metric == "multiple_choice_grade":
                likelihoods = results[:-1] if self._has_generative else results
                queries = self._doc_to_queries(doc)
                highest_score_index = _argmax(likelihoods)
                highest_score_key = queries[highest_score_index]
                res["multiple_choice_grade"] = int(self.doc_to_target(doc) == highest_score_key)
            elif metric == "exact_str_match":
                postprocessed = _postprocess_output(
                    results[-1],
                    max_length=self.max_length
                )
                res["exact_str_match"] = check_match(postprocessed, self.doc_to_target(doc))
            else:
                raise NotImplementedError(f"Metric {metric} isn't implemented")
        return res

    def aggregation(self):
        return {
            "multiple_choice_grade": mean,
            "exact_str_match": mean,
        }

    def higher_is_better(self):
        return {
            "multiple_choice_grade": True,
            "exact_str_match": True,
        }

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
            description = ""
        else:
            description = self._description
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + " " + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return description + labeled_examples + example



class disambiguation_qa(Task):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/disambiguation_qa/train.json"
    TEST_PATH = "/.../lm-evaluation-harness/dataset/disambiguation_qa/test.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH) 
        self._description = "Clarify the meaning of sentences with ambiguous pronouns.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n"
        #format options
        choice_dict = doc["target_scores"]
        choices = list(choice_dict.keys())
        formatted_options = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        formatted_options = f"Options:\n{formatted_options}\n"
        new_line = "\n"
        _input = "Sentence: " + doc['input']

        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "

        if pattern_id == 0:
            return f"{example_input_prefix}{prompt}{formatted_options}{_input}{new_line}{example_output_prefix}"
        elif pattern_id == 1:
            return f"{example_input_prefix}{prompt}{_input}{new_line}{formatted_options}{example_output_prefix}"
        elif pattern_id == 2:
            return f"{example_input_prefix}{_input}{new_line}{prompt}{formatted_options}{example_output_prefix}"

    def doc_to_target(self, doc):
        choices = list(doc["target_scores"].keys())
        return f"({chr(65 + choices.index(max(doc['target_scores'].items(), key=lambda x: x[1])[0]))})"

    def _doc_to_queries(self, doc):
        return list(doc["target_scores"].keys())

    def construct_requests(self, doc, ctx):
        requests = []
        if self._has_multi_choice:
            num_choices = len(doc["target_scores"])
            queries = [f"({chr(65 + i)})" for i in range(num_choices)]
            requests += [
                rf.loglikelihood(ctx, continuation)[0] for continuation in queries
            ]
        if self._has_generative:
            requests.append(
                rf.greedy_until(ctx, {"until": ["\nQ:","\n\nQ:","\n\n"," Q:"], "max_length": self.max_length}) 
            )
        return requests

    def process_results(self, doc, results):
        res = {}
        for metric in self.metrics:
            if metric == "multiple_choice_grade":
                likelihoods = results[:-1] if self._has_generative else results
                queries = self._doc_to_queries(doc)
                highest_score_index = _argmax(likelihoods)
                highest_score_key = queries[highest_score_index]
                res["multiple_choice_grade"] = doc["target_scores"][highest_score_key]
            elif metric == "exact_str_match":
                postprocessed = _postprocess_output(
                    results[-1]
                )
                res["exact_str_match"] = check_match(postprocessed, self.doc_to_target(doc))
            else:
                raise NotImplementedError(f"Metric {metric} isn't implemented")
        return res

    def aggregation(self):
        return {
            "multiple_choice_grade": mean,
            "exact_str_match": mean,
        }

    def higher_is_better(self):
        return {
            "multiple_choice_grade": True,
            "exact_str_match": True,
        }

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
            description = ""
        else:
            description = self._description
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + " " + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return description + labeled_examples + example



class sports_understanding(Task):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/sports_understanding/train.json"
    TEST_PATH = "/.../lm-evaluation-harness/dataset/sports_understanding/test.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH) 
        self._description = "Determine whether an artificially constructed sentence relating to sports is plausible or not.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt_templates = [
                "Is the following sentence plausible? {input}",
                "Is the following sentence {input} plausible?",
                "{input} Is the following sentence plausible?"
            ]
        prompt = prompt_templates[pattern_id].format(input=doc["input"])
        
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "\nA:" if num_fewshot != 0 else "\nA: "

        return f"{example_input_prefix}{prompt}{example_output_prefix}"

    def doc_to_target(self, doc):
        return doc["target"]

    def _doc_to_queries(self, doc):
        return ['yes', 'no']

    def construct_requests(self, doc, ctx):
        requests = []
        if self._has_multi_choice:
            queries = self._doc_to_queries(doc)
            requests += [
                rf.loglikelihood(ctx, continuation)[0] for continuation in queries
            ]
        if self._has_generative:
            requests.append(
                rf.greedy_until(ctx, {"until": ["\nQ:","\n\nQ:","\n\n"," Q:"], "max_length": self.max_length}) #
            )
        return requests

    def process_results(self, doc, results):
        res = {}
        for metric in self.metrics:
            if metric == "multiple_choice_grade":
                likelihoods = results[:-1] if self._has_generative else results
                queries = self._doc_to_queries(doc)
                highest_score_index = _argmax(likelihoods)
                highest_score_key = queries[highest_score_index]
                res["multiple_choice_grade"] = int(doc["target"] == highest_score_key)
            elif metric == "exact_str_match":
                postprocessed = _postprocess_output(
                    results[-1],
                    max_length=self.max_length
                )
                res["exact_str_match"] = check_match(postprocessed, self.doc_to_target(doc))
            else:
                raise NotImplementedError(f"Metric {metric} isn't implemented")
        return res

    def aggregation(self):
        return {
            "multiple_choice_grade": mean,
            "exact_str_match": mean,
        }

    def higher_is_better(self):
        return {
            "multiple_choice_grade": True,
            "exact_str_match": True,
        }
    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
            description = ""
        else:
            description = self._description
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + " " + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return description + labeled_examples + example



class navigate(Task):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/navigate/train.json"
    TEST_PATH = "/.../lm-evaluation-harness/dataset/navigate/test.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH)  
        self._description = "Given a series of navigation instructions, determine whether one would end up back at the starting point.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = " If you follow these instructions, do you return to the starting point?"
        prompt1 = "If you follow these instructions, do you return to the starting point? "
        prompt2 = "If you follow these instructions, do you return to the starting point?"
        choice = doc["options"]
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "

        if pattern_id == 0:
            return f"{example_input_prefix}{prompt2}{choice}{doc['input']}{new_line}{example_output_prefix}"
        elif pattern_id == 1:
            return f"{example_input_prefix}{prompt1}{doc['input']}{choice}{example_output_prefix}"
        elif pattern_id ==2:
            return f"{example_input_prefix}{doc['input']}{prompt}{choice}{example_output_prefix}" 

    def doc_to_target(self, doc):
        return doc["target"]

    def _doc_to_queries(self, doc):
        return ['Yes', 'No']

    def construct_requests(self, doc, ctx):
        requests = []
        if self._has_multi_choice:
            queries = self._doc_to_queries(doc)
            requests += [
                rf.loglikelihood(ctx, continuation)[0] for continuation in queries
            ]
        if self._has_generative:
            requests.append(
                rf.greedy_until(ctx, {"until": ["\nQ:","\n\nQ:","\n\n"," Q:"], "max_length": self.max_length}) 
            )
        return requests

    def process_results(self, doc, results):
        res = {}
        for metric in self.metrics:
            if metric == "multiple_choice_grade":
                likelihoods = results[:-1] if self._has_generative else results
                queries = self._doc_to_queries(doc)
                highest_score_index = _argmax(likelihoods)
                highest_score_key = queries[highest_score_index]
                res["multiple_choice_grade"] = int(self.doc_to_target(doc) == highest_score_key)
            elif metric == "exact_str_match":
                postprocessed = _postprocess_output(
                    results[-1],
                    max_length=self.max_length
                )
                res["exact_str_match"] = check_match(postprocessed, self.doc_to_target(doc))
            else:
                raise NotImplementedError(f"Metric {metric} isn't implemented")
        return res

    def aggregation(self):
        return {
            "multiple_choice_grade": mean,
            "exact_str_match": mean,
        }

    def higher_is_better(self):
        return {
            "multiple_choice_grade": True,
            "exact_str_match": True,
        }

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
            description = ""
        else:
            description = self._description
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + " " + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return description + labeled_examples + example



class logical_deduction_three_objects(Task):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/logical_deduction/train.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH)  
        self._description = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
        self.max_length = 128
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/logical_deduction/test_three.json"

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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = doc["prompt"]
        choice = doc["options"]
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "
        pattern_mapping = {
            0: f"{prompt}{choice}{doc['input']}{new_line}", 
            1: f"{prompt} {doc['input']}{choice}",
            2: f"{doc['input']} {prompt}{choice}" 
        }
        return f"{example_input_prefix}{pattern_mapping[pattern_id]}{example_output_prefix}"


    def doc_to_target(self, doc):
        return doc["target"]

    def _doc_to_queries(self, doc):
        options_str = doc["options"]
        matches = re.findall(r"\([A-Z]\)", options_str)
        return matches

    def construct_requests(self, doc, ctx):
        requests = []
        if self._has_multi_choice:
            queries = self._doc_to_queries(doc)
            requests += [
                rf.loglikelihood(ctx, continuation)[0] for continuation in queries
            ]
        if self._has_generative:
            requests.append(
                rf.greedy_until(ctx, {"until": ["\nQ:","\n\nQ:","\n\n"," Q:"], "max_length": self.max_length}) 
            )
        return requests

    def process_results(self, doc, results):
        res = {}
        for metric in self.metrics:
            if metric == "multiple_choice_grade":
                likelihoods = results[:-1] if self._has_generative else results
                queries = self._doc_to_queries(doc)
                highest_score_index = _argmax(likelihoods)
                highest_score_key = queries[highest_score_index]
                res["multiple_choice_grade"] = int(self.doc_to_target(doc) == highest_score_key)
            elif metric == "exact_str_match":
                postprocessed = _postprocess_output(
                    results[-1],
                    max_length=self.max_length
                )
                res["exact_str_match"] = check_match(postprocessed, self.doc_to_target(doc))
            else:
                raise NotImplementedError(f"Metric {metric} isn't implemented")
        return res

    def aggregation(self):
        return {
            "multiple_choice_grade": mean,
            "exact_str_match": mean,
        }

    def higher_is_better(self):
        return {
            "multiple_choice_grade": True,
            "exact_str_match": True,
        }

    def fewshot_context(self, doc, num_fewshot, pattern_id):
        if num_fewshot == 0:
            labeled_examples = ""
            description = ""
        else:
            description = self._description
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, num_fewshot, pattern_id) + " " + self.doc_to_target(doc)
                        for doc in self.fewshot_docs
                    ]
                )
                + "\n\n"
            )
        example = self.doc_to_text(doc, num_fewshot, pattern_id)
        return description + labeled_examples + example


class logical_deduction_five_objects(logical_deduction_three_objects):
    def __init__(self):
        super().__init__()
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/logical_deduction/test_five.json"

class logical_deduction_seven_objects(logical_deduction_three_objects):
    def __init__(self):
        super().__init__()
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/logical_deduction/test_seven.json"


class penguins_in_a_table(logical_deduction_three_objects):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/penguins_in_a_table/train.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH)  
        self._description = "Answer questions about a table of penguins and their attributes.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
        self.max_length = 128
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/penguins_in_a_table/test.json"

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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = doc["prompt"]
        _input = doc["input"]
        choice = doc["options"]
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "

        pattern_mapping = {
            0: f"{prompt}{choice}{doc['input']}{new_line}", 
            1: f"{prompt}  {doc['input']}{choice}",
            2: f"{doc['input']}  {prompt}{choice}"
        }
        return f"{example_input_prefix}{pattern_mapping[pattern_id]}{example_output_prefix}"

    def doc_to_target(self, doc):
        return doc["target"]


class salient_translation_error_detection(logical_deduction_three_objects):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/salient_translation_error_detection/train.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH)  
        self._description = "Detect the type of error in an English translation of a German source sentence.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
        self.max_length = 128
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/salient_translation_error_detection/test.json"

    def _load_json(self, file_path):
        with open(file_path,encoding='utf-8') as file:
            return json.load(file)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._load_json(self.TEST_PATH)

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = doc["prompt"]
        _input = doc["input"]
        choice = doc["options"]
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "
        pattern_mapping = {
            0: f"{prompt}{choice}{doc['input']}{new_line}", 
            1: f"{prompt}  {doc['input']}{choice}",
            2: f"{doc['input']}{choice}{prompt}{new_line}"
        }
        return f"{example_input_prefix}{pattern_mapping[pattern_id]}{example_output_prefix}"

    def doc_to_target(self, doc):
        return doc["target"]


class movie_recommendation(logical_deduction_three_objects):
    VERSION = 0
    TRAIN_PATH = "/.../lm-evaluation-harness/dataset/movie_recommendation/train.json"
    def __init__(self):
        self.fewshot_docs = self._load_json(self.TRAIN_PATH)  
        self._description = "Recommend movies similar to the given list of movies.\n\n"
        self.metrics = ["exact_str_match"] #exact_str_match
        self._has_multi_choice = "multiple_choice_grade" in self.metrics
        self._has_generative = "exact_str_match" in self.metrics
        self.max_length = 128
        self.TEST_PATH = "/.../lm-evaluation-harness/dataset/movie_recommendation/test.json"

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

    def doc_to_text(self, doc, num_fewshot, pattern_id):
        prompt = doc["prompt"]
        _input = doc["input"]
        choice = doc["options"]
        new_line = "\n"
        example_input_prefix = "Q: " if num_fewshot != 0 else "Q: "
        example_output_prefix = "A:" if num_fewshot != 0 else "A: "

        pattern_mapping = {
            0: f"{choice}{prompt} {doc['input']}{new_line}",
            1: f"{prompt} {doc['input']}{new_line}{choice}",
            2: f"{doc['input']} {prompt}{new_line}{choice}" 
        }
        return f"{example_input_prefix}{pattern_mapping[pattern_id]}{example_output_prefix}"

    def doc_to_target(self, doc):
        return doc["target"]


def check_match(generation, reference):
    postprocessed_generation = generation.startswith(reference.strip()) 
    return int(postprocessed_generation)

def _postprocess_output(text):
    if isinstance(text, list):
        return [
            _postprocess_output(mo)
            for mo in text
        ]
    # Ensure it is a string (will convert from bytes, ... as needed)
    if not isinstance(text, str):
        text = str(text, "utf-8")

    return text.strip()

def _argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(), 16) % len(
        max_indices
    )
    return max_indices[idx]


