"""Landscape primitives: filter normalization, slice/interpolation correctness and
that the model is always restored afterwards. Fast toy-MLP regression, CPU."""

from __future__ import annotations

import torch

from optbench.data.toy import build_toy_regression
from optbench.landscape.directions import filter_normalize, naive_normalize, random_direction
from optbench.landscape.params import get_params, set_params_
from optbench.landscape.surface import loss_interpolation_1d, loss_surface_2d, make_eval_loss
from optbench.models.toy_mlp import ToyMLP


def _unit():
    data = build_toy_regression(in_dim=8, n=300, batch_size=32, seed=0)
    model = ToyMLP(in_dim=8, hidden=16, out_dim=1, depth=2)
    eval_loss = make_eval_loss(model, data.val_loader, data.task, "toy_mlp")
    return model, eval_loss


def test_filter_normalize_matches_param_norm_per_tensor():
    model, _ = _unit()
    params = get_params(model)
    g = torch.Generator().manual_seed(0)
    d = filter_normalize(random_direction(params, g), params)
    for di, pi in zip(d, params):
        if pi.dim() >= 2:  # weight matrices: ||d_i|| == ||theta_i||
            assert abs(di.norm().item() - pi.norm().item()) < 1e-4
        else:  # biases: zeroed
            assert di.norm().item() == 0.0


def test_naive_normalize_matches_total_norm():
    model, _ = _unit()
    params = get_params(model)
    g = torch.Generator().manual_seed(1)
    d = naive_normalize(random_direction(params, g), params)
    total_d = torch.sqrt(sum((x.float() ** 2).sum() for x in d))
    total_p = torch.sqrt(sum((p.float() ** 2).sum() for p in params))
    assert abs(total_d.item() - total_p.item()) < 1e-3


def test_interpolation_endpoints_and_restore():
    model, eval_loss = _unit()
    theta_a = get_params(model)
    theta_b = [p + 0.3 * torch.randn_like(p) for p in theta_a]

    ts, ys = loss_interpolation_1d(model, eval_loss, theta_a, theta_b, [0.0, 1.0])
    set_params_(model, theta_a); la = eval_loss()
    set_params_(model, theta_b); lb = eval_loss()
    set_params_(model, theta_a)
    assert abs(ys[0] - la) < 1e-5
    assert abs(ys[1] - lb) < 1e-5
    # model restored to theta_a
    assert all(torch.equal(p.detach(), t) for p, t in zip(model.parameters(), theta_a))


def test_surface_center_equals_solution_and_restore():
    model, eval_loss = _unit()
    theta = get_params(model)
    g = torch.Generator().manual_seed(2)
    dx = filter_normalize(random_direction(theta, g), theta)
    dy = filter_normalize(random_direction(theta, g), theta)

    coords, Z = loss_surface_2d(model, eval_loss, theta, dx, dy, [-0.1, 0.0, 0.1])
    set_params_(model, theta); ref = eval_loss()
    assert Z.shape == (3, 3)
    assert abs(Z[1, 1] - ref) < 1e-5  # offset (0,0) == solution
    assert all(torch.equal(p.detach(), t) for p, t in zip(model.parameters(), theta))
