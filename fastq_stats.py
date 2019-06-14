"""Calculate stats for K-mer windows from FASTQ reads."""
from collections import defaultdict
from itertools import permutations, zip_longest
from typing import Iterable, List, Tuple

import numpy as np

from conf.const import BASES, FILENAME, SCALP_SIZE,\
    WILDCARD, WINDOW_SIZE


def grouper(read: Iterable, window_size: int) -> Iterable:
    """Group read into associated lines.
    :param read: Full read from FASTQ.
    :param window_size: Size of window.
    :return: Iterable of windows"""
    args = [iter(read)] * window_size
    return zip_longest(*args, fillvalue=None)


def scalp_reads(reads: List) -> List:
    """Truncate reads into given size.
    :param reads: List of read lines.
    :return Truncated list of reads"""
    return reads[:SCALP_SIZE]


def parse_fastq(filename: str) -> List:
    """Parse FASTQ file for calculations and scripting.
    :param filename: FASTQ file path.
    :return: List of associated read lines."""
    reads = []
    with open(filename, 'r') as f:
        for line in grouper(f, 4):
            reads.append(line)

    return scalp_reads(reads)


def generate_window(sequence: str, window_size: int) -> Iterable:
    """Generate K-mer windows.
    :param sequence: String sequence from read.
    :param window_size: Given window size.
    :return: Iterator of sliding windows."""
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i:i + window_size]


def align_reads_with_scores(reads: List[Tuple]) -> List[Tuple]:
    """Split read lines into K-mers.
    :param reads: List of all read lines.
    :return: Read lines split into K-mers."""
    short_reads = []
    for read in reads:
        clean_k_mer = read[1].rstrip('\n').replace('G', WILDCARD)
        clean_score = read[3].rstrip('\n')
        for k_mer, score \
                in zip(generate_window(clean_k_mer, WINDOW_SIZE),
                       generate_window(clean_score, WINDOW_SIZE)):
            short_reads.append((k_mer, score))

    return short_reads


def translate_scores(symbol: str) -> int:
    """Translate score byte based on Illumina 1.8 encoding.
    :param symbol: Score byte.
    :return: Translated integer score."""
    return ord(symbol) - 33


def insert_k_mer_permutations(possibilities: List[Tuple],
                              indices: List[int],
                              known: str) -> List[str]:
    """Re-insert K-mer possibilities from N base read.
    :param possibilities: List of permutations from unknown.
    :param indices: Indices where wildcard reads occur.
    :param known: Remaining reads that are not wildcards.
    :return: Exploded list of K-mer permutations."""
    k_mer_permutations = []
    for possibility in possibilities:
        possible_k_mer = known
        for index, letter in zip(indices, possibility):
            possible_k_mer = possible_k_mer[:index]\
                             + letter\
                             + possible_k_mer[index:]

        k_mer_permutations.append(possible_k_mer)

    return k_mer_permutations


def permutate_n_k_mers(k_mer: str) -> List[str]:
    """Find all possibilities for K-mers containing N.
    :param k_mer: K-mer with wildcard read.
    :return: List of all possible K-mers given wildcards."""
    n_positions = [index
                   for index, character
                   in enumerate(k_mer)
                   if character == WILDCARD]

    n_length = len(n_positions)
    known_bases = k_mer.replace(WILDCARD, '')
    possible_k_mer_n_values = list(permutations(BASES, n_length))

    return insert_k_mer_permutations(possible_k_mer_n_values,
                                     n_positions, known_bases)


def aggregate_reads_with_scores(read_score_windows: List[Tuple]):
    """Aggregate all permutated read windows with scores.
    :param read_score_windows: List of K-mers with associated scores.
    :return: Every read aggregated into K-mer with scores and wildcards permutated."""
    all_k_mers = []
    for read, score in read_score_windows:
        if WILDCARD in read:
            for possible_k_mer in permutate_n_k_mers(read):
                all_k_mers.append((possible_k_mer, score))
        else:
            all_k_mers.append((read, score))

    return all_k_mers


def find_most_abundant_k_mer(k_mers_with_scores: List[Tuple]):
    """Find the most abundant K-mer from reads.
    :param k_mers_with_scores: List of K-mers.
    :return: Pair of most abundant tuple and associated scores."""
    counted_k_mers = defaultdict(list)
    for k_mer, score in k_mers_with_scores:
        counted_k_mers[k_mer].append(score)

    max_k_mer = max(counted_k_mers, key=lambda x: len(counted_k_mers[x]))
    return max_k_mer, counted_k_mers[max_k_mer]


def calculate_k_mer_stats(scores: List[str]):
    """Calculate statistics of selected K-mer scores.
    :param scores: List of scores as bytes.
    :return: Group of quartile calculations."""
    average_quality_scores = []
    for score in scores:
        read_scores = [translate_scores(score_symbol) for score_symbol in score]
        average_quality_scores.append(sum(read_scores) / float(WINDOW_SIZE))

    quality_lower_quartile = np.percentile(average_quality_scores, 25)
    quality_median = np.percentile(average_quality_scores, 50)
    quality_upper_quartile = np.percentile(average_quality_scores, 75)

    return quality_lower_quartile, \
        quality_median, \
        quality_upper_quartile


if __name__ == '__main__':
    chunk_reads = parse_fastq(FILENAME)
    window_reads = align_reads_with_scores(chunk_reads)

    k_mer_aggregated_with_scores = \
        aggregate_reads_with_scores(window_reads)
    most_abundant_k_mer, k_mer_scores = \
        find_most_abundant_k_mer(k_mer_aggregated_with_scores)
    lower_quartile_score, median_quality_score, upper_quality_score = \
        calculate_k_mer_stats(k_mer_scores)

    print('#################### STATS ####################')
    print('Most abundant K-mer: {}'.format(most_abundant_k_mer))
    print('Count: {}'.format(len(k_mer_scores)))
    print('Quality lower quartile: {}'.format(lower_quartile_score))
    print('Quality median: {}'.format(median_quality_score))
    print('Quality upper quartile: {}'.format(upper_quality_score))
