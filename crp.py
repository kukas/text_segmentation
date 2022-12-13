import numba
import numpy as np
from collections import Counter
from numba import njit, types
from numba.typed import Dict
import random


@njit
def p_0(word, C, p_c):
    """Compute the base probability of a word."""
    assert len(word) > 0, "Word must be non-empty."
    return np.power(p_c / C, len(word)) * (1 - p_c)


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
    @njit
    def _step(text, s, counts, total, alpha, p_cont, C, p_c, T):
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
            joined_word = prev_word + next_word

            if s[i] == 0:
                # Remove joined word
                counts[joined_word] -= 1
                total -= 1
            else:
                # Remove separate words
                counts[prev_word] -= 1
                counts[next_word] -= 1
                total -= 2

            # Compute probabilities
            p[0] = (alpha * p_0(joined_word, C, p_c) + counts.get(joined_word, 0)) / (
                alpha + total
            )
            p[1] = (
                (alpha * p_0(prev_word, C, p_c) + counts.get(prev_word, 0))
                * (alpha * p_0(next_word, C, p_c) + counts.get(next_word, 0))
                / (alpha + total)
                / (alpha + total + 1)
                * p_cont
            )
            p = p / np.sum(p)
            s[i] = random.random() < p[1]

            if s[i] == 0:
                # Add joined word to counts
                counts[joined_word] = counts.get(joined_word, 0) + 1
                total += 1
            else:
                # Add separate words to counts
                counts[prev_word] = counts.get(prev_word, 0) + 1
                counts[next_word] = counts.get(next_word, 0) + 1
                total += 2

            # print(s, text)
            # print(prev_sep, i, next_sep)
            # print(prev_word, joined_word, next_word)

    def segmentation(self, text, numiter):
        """Segment text into sentences using CRP."""
        n = len(text)
        C = len(set(text))
        # initialize random binary array s_i for storing segmentation
        s = np.random.randint(2, size=n)
        counts, total = CRPTextSegmentation.compute_word_counts(text, s[1:])

        # convert counts into Numba typed dictionary
        counts_typed = Dict.empty(types.string, types.int64)
        for k, v in counts.items():
            counts_typed[k] = v

        for it in range(numiter):
            print("Iteration: ", it)
            CRPTextSegmentation._step(
                text,
                s,
                counts_typed,
                total,
                self.alpha,
                self.p_cont,
                C,
                self.p_c,
                self.T,
            )
            print(s, text, self._segment_text(text, s[1:]))


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
    assert _get_adjacent_separator(sep, 3, 1) == 5
    assert _get_adjacent_separator(sep, 3, -1) == 2
    assert _get_adjacent_separator(sep, 2, -1) == 0
    assert _get_adjacent_separator(sep, 5, -1) == 2
    assert _get_adjacent_separator(sep, 5, 1) == 8


if __name__ == "__main__":
    test_compute_word_counts()
    test_get_adjacent_separator()

    # open data_small.txt
    with open("data_small.txt", "r") as f:
        text = f.read()

    crp = CRPTextSegmentation(100, 0.99, 0.5, 1)
    crp.segmentation(text, 50)
