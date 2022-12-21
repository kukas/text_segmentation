import numba
import numpy as np
from collections import Counter
import time
from tqdm import tqdm
from numba import njit, types
from numba.typed import Dict
import random
import subprocess
import re
import logging

logger = logging.getLogger(__name__)


@njit
def compute_p_0(word_len, C, p_c):
    """Compute the base probability of a word."""
    # assert len(word) > 0, "Word must be non-empty."
    return np.power(p_c / C, word_len) * (1 - p_c)


@njit
def _get_adjacent_separator(s, i, direction):
    """Get adjacent separator in given direction."""
    adjacent = i + direction
    while adjacent > 0 and adjacent < len(s):
        if s[adjacent]:
            return adjacent
        adjacent += direction
    return adjacent


class CRPTextSegmentation:
    """Class implementing Chinese Restaurant Process (CRP) for text segmentation."""

    def __init__(self, alpha, p_cont, p_c, T):
        self.alpha = alpha
        self.p_cont = p_cont
        self.p_c = p_c
        self.T = T

    @staticmethod
    def _segment_text(text, segmentation):
        """Cut text into words using a segmentation."""
        assert len(text) == len(segmentation) + 1, "Segmentation is not valid."
        start = 0
        segmented_text = []
        for i in range(len(segmentation)):
            if segmentation[i]:
                word = text[start : i + 1]
                segmented_text.append(word)
                start = i + 1
        if start < len(text):
            word = text[start:]
            segmented_text.append(word)
        return segmented_text

    @staticmethod
    def compute_word_counts(text, segmentation):
        """Compute word counts for given segmentation of the text."""
        segmented_text = CRPTextSegmentation._segment_text(text, segmentation)
        counts = Counter(segmented_text)
        total = len(segmented_text)
        return counts, total

    @staticmethod
    def precompute_p_0(C, p_c):
        p_0 = []
        p_0.append(-1)
        for i in range(1, 1000):
            p_0.append(compute_p_0(i, C, p_c))
        p_0 = np.array(p_0)

        return p_0

    @staticmethod
    @njit
    def _step(text, s, counts, total, alpha, p_cont, C, p_c, p_0, T):
        """Perform one step of CRP."""
        n = len(s)
        random_perm = np.random.permutation(np.arange(1, n))
        p = np.zeros(2)

        # print(random_perm)
        for i in random_perm:
            prev_sep = _get_adjacent_separator(s, i, -1)
            next_sep = _get_adjacent_separator(s, i, 1)
            prev_word = text[prev_sep:i]
            next_word = text[i:next_sep]
            joined_word = text[prev_sep:next_sep]

            len_prev_word = i - prev_sep
            len_next_word = next_sep - i
            len_joined_word = next_sep - prev_sep
            # assert len_prev_word == len(prev_word)
            # assert len_next_word == len(next_word)
            # assert len_joined_word == len(joined_word)

            count_joined_word = counts.get(joined_word, 0)
            count_prev_word = counts.get(prev_word, 0)
            count_next_word = counts.get(next_word, 0)

            if s[i] == 0:
                # Remove joined word
                assert count_joined_word > 0, "Joined word not in counts."
                count_joined_word -= 1
                total -= 1
            else:
                # Remove separate words
                assert count_prev_word > 0, "Previous word not in counts."
                assert count_next_word > 0, "Next word not in counts."
                count_prev_word -= 1
                count_next_word -= 1
                # Gotcha! if the two words are the same
                # we need to decrease the count by 2 for both
                if prev_word == next_word:
                    count_prev_word -= 1
                    count_next_word -= 1
                total -= 2

            # Compute probabilities
            p[0] = (alpha * p_0[len_joined_word] + count_joined_word) / (alpha + total)
            p[1] = (
                (alpha * p_0[len_prev_word] + count_prev_word)
                * (alpha * p_0[len_next_word] + count_next_word)
                / (alpha + total)
                / (alpha + total + 1)
                * p_cont
            )
            p = p / (p[0] + p[1])
            if T != 1:
                p = p ** (1 / T)
                p = p / (p[0] + p[1])
            old_s = s[i]
            s[i] = random.random() < p[1]

            if s[i] == 0:
                # Add joined word to counts if it was split
                if old_s == 1:
                    counts[joined_word] = count_joined_word + 1
                    counts[prev_word] = count_prev_word
                    counts[next_word] = count_next_word
                total += 1
            else:
                # Add separate words to counts if they were joined
                if old_s == 0:
                    counts[joined_word] = count_joined_word
                    counts[prev_word] = count_prev_word + 1
                    counts[next_word] = counts.get(next_word, 0) + 1
                total += 2

        return total

        # print(s, text)
        # print(prev_sep, i, next_sep)
        # print(prev_word, joined_word, next_word)

    def segmentation(self, text, numiter, output_file=None, pre_step_callback=None):
        """Segment text into sentences using CRP."""
        start_time = time.time()
        n = len(text)
        C = len(set(text))
        # initialize random binary array s_i for storing segmentation
        s = np.random.randint(2, size=n)
        counts, total = CRPTextSegmentation.compute_word_counts(text, s[1:])

        # convert counts into Numba typed dictionary
        counts_typed = Dict.empty(types.string, types.int64)
        for k, v in counts.items():
            counts_typed[k] = v

        # precompute p_0
        p_0 = CRPTextSegmentation.precompute_p_0(C, self.p_c)

        results = []

        pbar = tqdm(range(numiter))
        for it in pbar:
            # print("Iteration: ", it)
            logger.info("Iteration: %d, Time: %f s", it, time.time() - start_time)
            start_time = time.time()

            if output_file is not None:
                segmented_text = CRPTextSegmentation._segment_text(text, s[1:])
                segmented_text = " ".join(segmented_text)
                output_file_formatted = output_file.format(it=it)
                with open(output_file_formatted, "w+") as f:
                    f.write(segmented_text)

                # run perl script to evaluate segmentation
                # extract precision, recall and F1 score from the stdout
                process = subprocess.run(
                    ["perl", "eval.pl", "data_small_gold.txt", output_file_formatted],
                    capture_output=True,
                )
                stdout = process.stdout.decode("utf-8")

                # parse out the precision, recall and F1 score from the stdout using regex
                # format is: P:0.017, R:0.044, F:0.024
                precision, recall, f1 = map(float, re.findall(r"\d+\.\d+", stdout))
                # os.system(f"perl eval.pl data_small_gold.txt {output_file_formatted}")
                logger.info("P: %f, R: %f, F: %f", precision, recall, f1)
                pbar.set_postfix({"P": precision, "R": recall, "F": f1})

                result = {
                    "iteration": it,
                    "segmentation": segmented_text,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "p_c": self.p_c,
                    "alpha": self.alpha,
                    "T": self.T,
                }

            if pre_step_callback is not None:
                pre_step_callback(self, it, result)
            # print(s, text, self._segment_text(text, s[1:]))
            results.append(result)

            total = CRPTextSegmentation._step(
                text,
                s,
                counts_typed,
                total,
                self.alpha,
                self.p_cont,
                C,
                self.p_c,
                p_0,
                self.T,
            )

        return results


def test_compute_word_counts():
    """Test compute_word_counts function."""
    text = "abcd"
    segmentation = np.array([0, 0, 0])
    counts, total = CRPTextSegmentation.compute_word_counts(text, segmentation)
    assert dict(counts) == {"abcd": 1}

    text = "abcd"
    segmentation = np.array([1, 0, 1])
    counts, total = CRPTextSegmentation.compute_word_counts(text, segmentation)
    assert dict(counts) == {"a": 1, "bc": 1, "d": 1}

    text = "aaabbbcccddd"
    segmentation = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    counts, total = CRPTextSegmentation.compute_word_counts(text, segmentation)
    assert dict(counts) == {"aaa": 1, "bbb": 1, "ccc": 1, "ddd": 1}
    assert total == 4

    text = "Tributes poured in from around the world Thursday to the late Labour Party leader John Smith"
    text_without_spaces = text.replace(" ", "")
    segmentation = np.zeros(len(text_without_spaces) - 1)
    i = 0
    for letter in text:
        if letter == " ":
            segmentation[i - 1] = 1
        else:
            i += 1
    counts, total = CRPTextSegmentation.compute_word_counts(
        text_without_spaces, segmentation
    )
    assert counts["Tributes"] == 1
    assert counts["around"] == 1
    assert counts["the"] == 2
    assert total == 16


def test_get_adjacent_separator():
    sep = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    assert _get_adjacent_separator(sep, 1, -1) == 0
    assert _get_adjacent_separator(sep, 1, 1) == 2
    assert _get_adjacent_separator(sep, 3, 1) == 5
    assert _get_adjacent_separator(sep, 3, -1) == 2
    assert _get_adjacent_separator(sep, 2, -1) == 0
    assert _get_adjacent_separator(sep, 5, -1) == 2
    assert _get_adjacent_separator(sep, 5, 1) == 8


def test_step():
    p_c = 0.5
    alpha = 100
    p_cont = 0.99
    T = 1
    numiter = 10

    text = "aaaabbbbbccccdddd"
    n = len(text)
    C = len(set(text))
    # initialize random binary array s_i for storing segmentation
    s = np.random.randint(2, size=n)
    counts, total = CRPTextSegmentation.compute_word_counts(text, s[1:])

    p_0 = CRPTextSegmentation.precompute_p_0(C, p_c)

    pbar = tqdm(range(numiter))
    for it in pbar:
        segmented_text = CRPTextSegmentation._segment_text(text, s[1:])
        segmented_text = " ".join(segmented_text)
        print(segmented_text)
        print(counts, total)

        # print("=== step begin ===")
        total = CRPTextSegmentation._step(
            text,
            s,
            counts,
            total,
            alpha,
            p_cont,
            C,
            p_c,
            p_0,
            T,
        )
        # print("=== step end ===")
        segmented_text = CRPTextSegmentation._segment_text(text, s[1:])
        segmented_text = " ".join(segmented_text)
        # print(segmented_text)
        # print(counts, total)
        check_counts, check_total = CRPTextSegmentation.compute_word_counts(text, s[1:])
        assert total == check_total
        for word, count in counts.items():
            assert count == check_counts[word]
        # print()
        # assert (counts, total) == CRPTextSegmentation.compute_word_counts(text, s[1:])


if __name__ == "__main__":
    test_compute_word_counts()
    test_get_adjacent_separator()
    # test_step()

    # open data_small.txt
    with open("data_small.txt", "r") as f:
        text = f.read()

    crp = CRPTextSegmentation(100, 0.99, 0.5, 1)
    crp.segmentation(text, 20, output_file="output_{it}.txt")
