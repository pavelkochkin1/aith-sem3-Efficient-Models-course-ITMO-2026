import math

import pytest
import torch
from torch.nn import functional as F

from hw2.initialization import hadamard, zero_init_
from hw2.models import MLP, PatchTransformer
from hw2.train import gradient_diagnostics, spectrum_metrics


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_square_and_contracting_are_partial_identity(shape):
    for normalization in ("paper", "author"):
        weight = torch.empty(shape, dtype=torch.float64)
        zero_init_(weight, normalization)
        assert torch.equal(weight, torch.eye(*shape, dtype=torch.float64))


@pytest.mark.parametrize("rows,columns", [(4, 3), (7, 3), (16, 5)])
def test_expansion_is_correctly_truncated_and_scaled(rows, columns):
    order = 1 << (rows - 1).bit_length()
    paper = zero_init_(torch.empty(rows, columns, dtype=torch.float64), "paper")
    author = zero_init_(torch.empty_like(paper), "author")
    expected = hadamard(order, dtype=torch.float64)[:rows, :columns] / math.sqrt(order)
    torch.testing.assert_close(author, expected)
    torch.testing.assert_close(paper, math.sqrt(2) * author)


def test_hadamard_orthogonality_and_two_normalization_conventions():
    h = hadamard(16, dtype=torch.float64)
    torch.testing.assert_close(h.T @ h, 16 * torch.eye(16, dtype=h.dtype))
    for normalization, norm in (("paper", 2), ("author", 1)):
        weight = zero_init_(torch.empty(16, 5, dtype=torch.float64), normalization)
        torch.testing.assert_close(weight.T @ weight, norm * torch.eye(5, dtype=weight.dtype))


@pytest.mark.parametrize("method", ["zero_paper", "zero_author", "identity", "zero_all"])
def test_deterministic_methods_ignore_seed(method):
    torch.manual_seed(1)
    first = MLP(4, 8, method).state_dict()
    torch.manual_seed(99)
    second = MLP(4, 8, method).state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_partial_identity_extra_directions_stay_dead_under_sgd():
    torch.manual_seed(3)
    model = MLP(4, 16, "identity")
    inputs = torch.rand(32, 4)
    targets = torch.randint(10, (32,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(8):
        optimizer.zero_grad()
        F.cross_entropy(model(inputs), targets).backward()
        optimizer.step()
    assert torch.count_nonzero(model.first.weight[4:]) == 0
    assert torch.count_nonzero(model.features(inputs)[:, 4:]) == 0
    residual = model.hidden.weight.detach() - torch.eye(16)
    assert torch.count_nonzero(residual[:, 4:]) == 0
    assert spectrum_metrics(residual, [1e-4])["rank_tol_0.0001"] <= 4


def test_completely_zero_relu_mlp_has_no_parameter_gradient():
    model = MLP(4, 16, "zero_all")
    F.cross_entropy(model(torch.randn(32, 4)), torch.randint(10, (32,))).backward()
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters())


def test_corrected_hadamard_relu_span_example():
    # Appendix A.4's four +/- coordinate inputs span only rank 3 after ReLU.
    weight = zero_init_(torch.empty(4, 3, dtype=torch.float64), "paper")
    coordinate_inputs = torch.tensor([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=weight.dtype)
    diagonal_inputs = torch.tensor([[0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=weight.dtype)
    assert torch.linalg.matrix_rank(F.relu(coordinate_inputs @ weight.T)).item() == 3
    assert torch.linalg.matrix_rank(F.relu(diagonal_inputs @ weight.T)).item() == 4


def test_zero_inspired_attention_starts_uniform_with_staged_gradients():
    torch.manual_seed(2)
    model = PatchTransformer(width=16, heads=4, depth=1, ff_width=32, method="zero_inspired")
    images, targets = torch.rand(8, 1, 28, 28), torch.randint(10, (8,))
    diagnostic = gradient_diagnostics(model, images, targets, 0)
    gradients = diagnostic["gradient_norms"]
    assert gradients["blocks.0.attention.q.weight"] == 0
    assert gradients["blocks.0.attention.k.weight"] == 0
    assert gradients["blocks.0.attention.v.weight"] > 0
    assert gradients["blocks.0.attention.out.weight"] == 0
    assert diagnostic["attention"][0]["head_attention_difference"] == 0
    assert diagnostic["attention"][0]["attention_entropy"] == pytest.approx(math.log(16), abs=1e-6)


def test_stable_rank_differs_from_numerical_rank():
    matrix = torch.diag(torch.tensor([1, 1e-3, 0]))
    metrics = spectrum_metrics(matrix, [1e-4, 1e-2])
    assert metrics["rank_tol_0.0001"] == 2
    assert metrics["rank_tol_0.01"] == 1
    assert metrics["stable_rank"] == pytest.approx(1 + 1e-6)
