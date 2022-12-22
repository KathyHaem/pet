# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import csv
import os
from typing import List

from pet.task_helpers import MultiMaskTaskHelper
from pet.tasks import DataProcessor, PROCESSORS, TASK_HELPERS, GenerativeDataProcessor, UNLABELED_SET, DEV_SET, \
    TEST_SET, TRAIN_SET
from pet.utils import InputExample, GenerativeInputExample


class MorphoSubjectObjectDataProcessor(GenerativeDataProcessor):
    # Set this to the name of the task
    TASK_NAME = "morpho-subj-obj"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.tsv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.tsv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "eval.tsv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.tsv"

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = -1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 2

    def get_dev_examples(self, data_dir) -> List[GenerativeInputExample]:
        return self._create_examples(os.path.join(data_dir, MorphoSubjectObjectDataProcessor.DEV_FILE_NAME), DEV_SET)

    def get_test_examples(self, data_dir) -> List[GenerativeInputExample]:
        return self._create_examples(os.path.join(data_dir, MorphoSubjectObjectDataProcessor.TEST_FILE_NAME), TEST_SET)

    def get_unlabeled_examples(self, data_dir) -> List[GenerativeInputExample]:
        return self._create_examples(
            os.path.join(data_dir, MorphoSubjectObjectDataProcessor.UNLABELED_FILE_NAME), UNLABELED_SET)

    def get_train_examples(self, data_dir) -> List[GenerativeInputExample]:
        return self._create_examples(
            os.path.join(data_dir, MorphoSubjectObjectDataProcessor.TRAIN_FILE_NAME), TRAIN_SET)

    @staticmethod
    def _create_examples(path, set_type) -> List[GenerativeInputExample]:
        """Creates examples for the training and dev sets."""
        examples = []

        with open(path, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter='\t')
            for idx, row in enumerate(reader):
                guid = "%s-%s" % (set_type, idx)
                output_text = None if set_type == UNLABELED_SET else row[MorphoSubjectObjectDataProcessor.LABEL_COLUMN]
                text_a = row[MorphoSubjectObjectDataProcessor.TEXT_A_COLUMN]
                text_b = None
                example = GenerativeInputExample(guid=guid, text_a=text_a, text_b=text_b, output_text=output_text)
                examples.append(example)

        return examples


# register the processor for this task with its name
PROCESSORS[MorphoSubjectObjectDataProcessor.TASK_NAME] = MorphoSubjectObjectDataProcessor


class MorphoBinaryClassDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "morpho-class-binary"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.tsv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.tsv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "eval.tsv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.tsv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["YES", "NO"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = -1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 2

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, MorphoBinaryClassDataProcessor.TRAIN_FILE_NAME), TRAIN_SET)

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, MorphoBinaryClassDataProcessor.DEV_FILE_NAME), DEV_SET)

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        return self._create_examples(os.path.join(data_dir, MorphoBinaryClassDataProcessor.TEST_FILE_NAME), TEST_SET)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        filename = os.path.join(data_dir, MorphoBinaryClassDataProcessor.UNLABELED_FILE_NAME)
        if os.path.isfile(filename):
            return self._create_examples(filename, UNLABELED_SET)
        return []

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MorphoBinaryClassDataProcessor.LABELS

    @staticmethod
    def _create_examples(path, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        with open(path, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter='\t')
            for idx, row in enumerate(reader):
                guid = "%s-%s" % (set_type, idx)
                label = row[MorphoBinaryClassDataProcessor.LABEL_COLUMN]
                text_a = row[MorphoBinaryClassDataProcessor.TEXT_A_COLUMN]
                text_b = None
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


# register the processor for this task with its name
PROCESSORS[MorphoBinaryClassDataProcessor.TASK_NAME] = MorphoBinaryClassDataProcessor

# optional: if you have to use verbalizers that correspond to multiple tokens, uncomment the following line
# TASK_HELPERS[MyTaskDataProcessor.TASK_NAME] = MultiMaskTaskHelper
