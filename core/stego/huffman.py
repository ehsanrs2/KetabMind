"""Equal-length Huffman codeword substitution utilities.

This module provides functions to embed and extract binary messages from a
sequence of Huffman codewords by swapping pairs of equal-length codewords.
The mapping table defines *safe* pairs of codewords that share the same bit
length, ensuring that frame sizes remain unchanged after embedding.
"""
from __future__ import annotations

from collections.abc import Sequence

# Mapping table of equal-length Huffman codeword pairs.
# Each key represents the codeword used to encode a message bit ``0`` and
# maps to the codeword representing bit ``1``. All pairs must be of equal
# length to maintain frame size.
MAPPING_TABLE: dict[str, str] = {
    "000": "001",
    "010": "011",
    "100": "101",
    "110": "111",
}

# Reverse mapping for quick lookups during embedding and extraction.
_REVERSE_MAPPING: dict[str, str] = {v: k for k, v in MAPPING_TABLE.items()}
_ALL_CODES = set(MAPPING_TABLE) | set(_REVERSE_MAPPING)


def findCandidates(codewords: Sequence[str]) -> list[int]:  # noqa: N802
    """Return indices of codewords eligible for substitution.

    A codeword is considered a candidate if it exists in the mapping table
    either as a ``0`` or ``1`` representation. The function returns the
    indices of all such codewords within the provided sequence.
    """

    return [i for i, cw in enumerate(codewords) if cw in _ALL_CODES]


def applyEmbedding(codewords: Sequence[str], message_bits: str) -> list[str]:  # noqa: N802
    """Embed ``message_bits`` into ``codewords`` using substitution.

    Parameters
    ----------
    codewords:
        Original sequence of Huffman codewords.
    message_bits:
        Binary string representing the message to embed.

    Returns
    -------
    List[str]
        New list of codewords with the message embedded.

    Raises
    ------
    ValueError
        If ``message_bits`` is longer than the number of available
        substitution candidates.
    """

    candidates = findCandidates(codewords)
    if len(message_bits) > len(candidates):
        msg = "Message too long for available candidates"
        raise ValueError(msg)

    result = list(codewords)
    for bit, idx in zip(message_bits, candidates, strict=False):
        current = result[idx]
        if bit == "1":
            # Swap to the paired codeword if we hold a ``0`` representation.
            if current in MAPPING_TABLE:
                result[idx] = MAPPING_TABLE[current]
        else:  # bit == "0"
            # Ensure the codeword represents ``0``.
            if current in _REVERSE_MAPPING:
                result[idx] = _REVERSE_MAPPING[current]
    return result


def extractMessage(codewords: Sequence[str], candidate_indices: Sequence[int]) -> str:  # noqa: N802
    """Extract an embedded message from ``codewords``.

    Parameters
    ----------
    codewords:
        Sequence containing the (possibly modified) Huffman codewords.
    candidate_indices:
        Indices where substitutions may have occurred, typically the output of
        :func:`findCandidates`.

    Returns
    -------
    str
        Recovered binary message.
    """

    bits: list[str] = []
    for idx in candidate_indices:
        cw = codewords[idx]
        if cw in MAPPING_TABLE:
            bits.append("0")
        elif cw in _REVERSE_MAPPING:
            bits.append("1")
    return "".join(bits)
