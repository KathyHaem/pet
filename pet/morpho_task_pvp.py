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
be added. The PVP is responsible for applying patterns to inputs and mapping
labels to their verbalizations (see the paper for more details on PVPs).
This file shows an example of a PVP for a new task.
"""

from typing import List, Tuple, Optional

from pet.pvp import PVP, PVPS, GenerativePVP, FilledPattern
from pet.tasks import METRICS
from pet.utils import InputExample, get_verbalization_ids
from transformers import PreTrainedTokenizer, GPT2Tokenizer


class MorphoSubjectObjectPVP(GenerativePVP):
    METRICS = ["rouge1", "rouge2", "rougeL"]
    TASK_NAME = "morpho-subj-obj"

    def generative_prefix(self) -> Optional[str]:
        # todo we may want one?? but idk what it would be so far
        return None

    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.

        For now we just have one kind of pattern: Using the already-generated question that Marion gave me.
        However, these do correspond to several variants.
        """
        questions = example.text_a.split(" ||| ")

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id <= len(questions):
            text_a = questions[self.pattern_id]
            prefix = [self.generative_prefix()] if self.no_decoder_prefix and self.generative_prefix() else []
            # We tell the tokenizer that text_a can be truncated if the resulting sequence is longer than
            # our language model's max sequence length
            text_a = self.shortenable(text_a)

            # this corresponds to the pattern text_a [MASK]
            return prefix + [text_a, self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


PVPS[MorphoSubjectObjectPVP.TASK_NAME] = MorphoSubjectObjectPVP
METRICS[MorphoSubjectObjectPVP.TASK_NAME] = MorphoSubjectObjectPVP.METRICS


class MorphoBinaryClassPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "morpho-class-binary"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER = {
        "YES": ["Yes"],
        "NO": ["No"]
    }

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.

        For now we just have one kind of pattern: Using the already-generated question that Marion gave me.
        However, these do correspond to several variants.
        """
        questions = example.text_a.split(" ||| ")

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id <= len(questions):
            text_a = questions[self.pattern_id]
            # We tell the tokenizer that text_a can be truncated if the resulting sequence is longer than
            # our language model's max sequence length
            text_a = self.shortenable(text_a)

            # this corresponds to the pattern text_a [MASK]
            return [text_a, self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return MorphoBinaryClassPVP.VERBALIZER[label]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        # here is the main difference to the overridden method (is_pretokenized).
        # though tbh during debugging i'm not even sure if that makes a difference??
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, is_pretokenized=True, **kwargs), s) for x, s in
                   parts_a if x]
        # i'm also not even bothering to process parts_b because so far i'm quite confident we won't have any (!)
        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = None

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                mask_idx = input_ids.index(self.mask_id)
                assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
                assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids


# register the PVP for this task with its name
PVPS[MorphoBinaryClassPVP.TASK_NAME] = MorphoBinaryClassPVP
