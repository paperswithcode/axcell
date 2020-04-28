#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import Counter


def get_probs(occurrences):
    """
    Computes conditional probabilities based on frequency of co-occurrences

    Parameters
    ----------
    occurrences: occurences[x][y] number of times with (X=x and Y=y)

    Returns
    -------
    probs : probs[x][y] = Pr(Y=y | X=x)
    reverse_probs : reverse_probs[y][x] = Pr(X=x | Y=y)
    """
    probs = {}
    reverse_probs = {}
    y_occ = Counter()
    for x, ys in occurrences.items():
        total = sum(ys.values())
        probs[x] = {}
        for y, occ in ys.items():
            probs[x][y] = occ / total
            y_occ[y] += occ
    for x, ys in occurrences.items():
        for y, occ in ys.items():
            reverse_probs.setdefault(y, {})[x] = occ / y_occ[y]

    return probs, reverse_probs


def reverse_probs(probs):
    """
    Reverses the conditional probability assuming that variables are uniformly distributed

    Parameters
    ----------
    probs : probs[x][y] = Pr(Y=y | X=x)

    Returns
    -------
    reverse : reverse[y][x] = Pr(X=x | Y=y) assuming X is uniform
    """
    reverse = {}
    for x, probs_x in probs.items():
        for y, p in probs_x.items():
            reverse.setdefault(y, {})[x] = p
    for y, probs_y in reverse.items():
        norm = sum(probs_y.values())
        for x, p in probs_y.items():
            probs_y[x] = p / norm
    return reverse
