"""Tests for equal-length Huffman codeword substitution."""
from core.stego.huffman import applyEmbedding, extractMessage, findCandidates


def test_round_trip_and_length_preservation():
    """Embedding and extraction should be lossless and length-safe."""
    # Synthetic frame comprised solely of codewords from the mapping table.
    frame = ["000", "010", "100", "110"]
    message = "1010"

    candidates = findCandidates(frame)
    embedded = applyEmbedding(frame, message)

    # Frame length must remain unchanged because substitutions are equal-length.
    assert sum(len(cw) for cw in frame) == sum(len(cw) for cw in embedded)

    extracted = extractMessage(embedded, candidates)
    assert extracted == message
