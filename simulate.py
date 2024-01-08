#
# (c) 2024 Takahiro Hashimoto
#
import argparse
import copy
import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class SandGrid:
    def __init__(self, grid_size: int, low: float = 2.85, high: float = 3.15) -> None:
        self.size = grid_size
        self.data = np.random.uniform(low, high, (grid_size, grid_size))  # Initialize with random height

    def __getitem__(self, index: tuple[int, int]) -> float:
        i, j = index
        return self.data[i % self.size, j % self.size]  # Periodic boundary condition

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        i, j = index
        self.data[i % self.size, j % self.size] = value  # Periodic boundary condition


class SandMovieWriter:
    def __init__(self, wind_speed: float, interval: int = -1) -> None:
        self.fig, self.ax = plt.subplots()
        self.has_plotted = False
        self.wind_speed = wind_speed
        self.interval = interval
        self.frames = []
        self.count = 0  # Count number of plots

    def plot(self, grid: SandGrid, vmin: float = -3.0, vmax: float = 12.0) -> None:
        self.count += 1
        im = self.ax.imshow(grid.data.T, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")  # ~2 ms overhead
        if not self.has_plotted:
            self.ax.set_title(f"Sand pattern. wind speed: {self.wind_speed:.1f}")
            self.fig.colorbar(im, ax=self.ax)
            self.fig.tight_layout()
            self.has_plotted = True
        self.frames.append([im])

        # Save figure at intervals for debug (very slow)
        if self.interval > 0:
            if self.count == 1 or self.count % self.interval == 0:
                os.makedirs("figure", exist_ok=True)
                self.fig.savefig(f"figure/{self.count}.png")

    def save_movie(self, output_path: str) -> None:
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=33)
        ani.save(output_path)


def apply_jumping_rule(
    grid: SandGrid, wind_speed: float = 3.0, jump_factor: float = 1.0, sand_quantity: float = 0.1
) -> SandGrid:
    for _ in range(grid.size * grid.size):
        i, j = random.randint(0, grid.size - 1), random.randint(0, grid.size - 1)
        height = grid[i, j]
        jump_length = int(wind_speed + jump_factor * height)
        grid[i, j] -= sand_quantity
        grid[i + jump_length, j] += sand_quantity
    return grid


def apply_rolling_rule(grid: SandGrid, diffusion_coef: float = 0.1) -> SandGrid:
    new_grid = copy.deepcopy(grid)
    for i in range(grid.size):
        for j in range(grid.size):
            new_grid[i, j] = (
                (1 - diffusion_coef) * grid[i, j]
                + diffusion_coef / 6 * (grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] + grid[i, j + 1])
                + diffusion_coef / 12 * (grid[i + 1, j + 1] + grid[i - 1, j - 1] + grid[i + 1, j - 1] + grid[i - 1, j + 1])
            )
    return new_grid


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sand pattern simulation")
    parser.add_argument("-w", "--wind_speed", type=float, metavar="V", default=3.0, help="wind speed (default: 3.0)")
    parser.add_argument("-g", "--grid_size", type=int, metavar="N", default=100, help="grid size (default: 100)")
    parser.add_argument("-i", "--num_iter", type=int, metavar="N", default=500, help="# iterations (default: 500)")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    grid = SandGrid(args.grid_size)  # Sand height
    writer = SandMovieWriter(args.wind_speed)

    # Simulate
    # NOTE: 27 fps, 100x100 grid, i7-12700H
    for _ in tqdm(range(args.num_iter)):
        grid = apply_jumping_rule(grid, args.wind_speed)
        grid = apply_rolling_rule(grid)
        writer.plot(grid)

    # Save as movie
    # NOTE: 14 fps, same condition
    print("Writing movie...")
    os.makedirs("movie", exist_ok=True)
    output_path = f"movie/sand_pattern_w{int(10 * args.wind_speed)}_g{args.grid_size}_i{args.num_iter}.mp4"
    writer.save_movie(output_path)


if __name__ == "__main__":
    main()
