import torch


def conv_pad(input: torch.Tensor, weight: torch.Tensor, padding=1) -> torch.Tensor:
    input = torch.nn.functional.pad(
        input, (padding, padding, padding, padding), "circular"
    )
    input = torch.nn.functional.conv2d(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    )
    return input


def min_max(input: torch.Tensor) -> torch.Tensor:
    return (input - input.min()) / (input.max() - input.min())


def x_or_y(x: torch.Tensor, y: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    # Get angle from x, y and do put into equation from Malus's law
    angle = torch.atan2(y, x)
    return torch.rand(x.shape, device=device) > torch.cos(angle) ** 2


def make_kernel(x_mul=1.0, y_mul=1.0, pow=1.0, swap=False, device="cuda"):
    grid = torch.tensor([-1.0, 0.0, 1.0], device=device)
    grid_x, grid_y = torch.meshgrid(grid, grid)
    d = (grid_x ** 2 + grid_y ** 2).sqrt()
    d = torch.where(torch.isclose(d, torch.zeros(())), d.new_zeros(()), 1.0 / d)
    grid_x = grid_x * d
    grid_y = grid_y * d
    grid_x = x_mul * grid_x / grid_x.abs().sum() ** pow
    grid_y = y_mul * grid_y / grid_y.abs().sum() ** pow
    if swap:
        kernel = torch.stack([grid_x, grid_y], dim=0).reshape(2, 1, 3, 3)
    else:
        kernel = torch.stack([grid_y, grid_x], dim=0).reshape(2, 1, 3, 3)
    return kernel.to(device)
