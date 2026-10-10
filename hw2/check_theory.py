"""Check the four-vector example in ZerO v3, Lemma 2 / Appendix A.4."""

import numpy as np


def check_example():
    h = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    coordinate_inputs = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    diagonal_inputs = np.array([[0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]])
    original = np.maximum(0, coordinate_inputs @ h[:, :3].T)
    corrected = np.maximum(0, diagonal_inputs @ h[:, :3].T)
    paper_rank = int(np.linalg.matrix_rank(original))
    corrected_rank = int(np.linalg.matrix_rank(corrected))
    assert np.array_equal(original[0] + original[1], original[2] + original[3])
    assert paper_rank == 3 and corrected_rank == 4
    return {"paper_features": original, "corrected_features": corrected,
            "paper_observed_rank": paper_rank, "corrected_rank": corrected_rank}


def main():
    result = check_example()
    print(f"Исходные четыре признака, rank={result['paper_observed_rank']}:\n", result["paper_features"])
    print(f"Исправленные четыре признака, rank={result['corrected_rank']}:\n", result["corrected_features"])


if __name__ == "__main__":
    main()
