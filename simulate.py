#
# (c) 2024 Takahiro Hashimoto
#
import argparse
import copy
import os
import random
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class SandGrid:
    def __init__(self, grid_size: int, low: float, high: float) -> None:
        self.x_size = grid_size
        self.y_size = grid_size
        self.data = np.random.uniform(low, high, (self.x_size, self.y_size))  # Initialize with random height

    def __getitem__(self, index: tuple[int, int]) -> float:
        i, j = index
        return self.data[i % self.x_size, j % self.y_size]  # Periodic boundary condition

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        i, j = index
        self.data[i % self.x_size, j % self.y_size] = value  # Periodic boundary condition

    def partial_x(self, i: int, j: int) -> float:
        """Partial derivative for x direction with periodic boundary condition."""
        return (self[i + 1, j] - self[i - 1, j]) / 2

    def diffusion_component(self) -> np.ndarray:
        first_adjacent_cells = (
            np.roll(self.data, -1, axis=0)
            + np.roll(self.data, 1, axis=0)
            + np.roll(self.data, -1, axis=1)
            + np.roll(self.data, 1, axis=1)
        )
        second_adjacent_cells = (
            np.roll(self.data, (-1, -1), axis=(0, 1))
            + np.roll(self.data, (-1, 1), axis=(0, 1))
            + np.roll(self.data, (1, -1), axis=(0, 1))
            + np.roll(self.data, (1, 1), axis=(0, 1))
        )
        return first_adjacent_cells / 6 + second_adjacent_cells / 12


class SandMovieWriter:
    def __init__(
        self, title_str: str, z_min: float = -3.0, z_max: float = 12.0, interval: int = 1, save_image: bool = False
    ) -> None:
        self.title_str = title_str
        self.z_min = z_min
        self.z_max = z_max
        self.interval = interval
        self.save_image = save_image  # Save figure for debug (very slow)

        self.fig, self.ax = plt.subplots()
        self.count = -1  # Number of plot() calls
        self.frames: list = []

    def plot(self, grid: SandGrid) -> None:
        self.count += 1
        if self.count % self.interval == 0:
            im = self.ax.imshow(grid.data.T, cmap="viridis", vmin=self.z_min, vmax=self.z_max, origin="lower")  # ~2 ms
            if self.count == 0:
                self.ax.set_title(self.title_str)
                self.fig.colorbar(im, ax=self.ax)
                self.fig.tight_layout()
            self.frames.append([im])

            if self.save_image:
                self.fig.savefig("sand_pattern.png")

    def save_movie(self, output_path: str) -> None:
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=33)
        ani.save(output_path)


class SandSimulator:
    def __init__(self, grid_size: int, low: float, high: float) -> None:
        self.grid = SandGrid(grid_size, low=low, high=high)

    def step(self) -> None:
        pass

    def apply_rolling_rule(self, diffusion_coef: float) -> None:
        g = self.grid
        new_grid = copy.deepcopy(g)

        if True:  # Use numpy.roll for faster computation
            new_grid.data = (1 - diffusion_coef) * g.data + diffusion_coef * g.diffusion_component()
        else:
            for i in range(g.x_size):
                for j in range(g.y_size):
                    new_grid[i, j] = (
                        (1 - diffusion_coef) * g[i, j]
                        + diffusion_coef / 6 * (g[i - 1, j] + g[i + 1, j] + g[i, j - 1] + g[i, j + 1])
                        + diffusion_coef / 12 * (g[i + 1, j + 1] + g[i - 1, j - 1] + g[i + 1, j - 1] + g[i - 1, j + 1])
                    )
        self.grid = new_grid

    def calc_stats(self) -> str:
        stats_str = (
            f"min: {self.grid.data.min():4.1f}, max: {self.grid.data.max():4.1f}, mean: {self.grid.data.mean():4.2f}, "
            f"std: {self.grid.data.std():4.2f}"
        )
        return stats_str


class SandPatternSimulator(SandSimulator):
    def __init__(self, grid_size: int = 100, wind_speed: float = 3.0, low: float = 2.85, high: float = 3.15) -> None:
        super().__init__(grid_size, low=low, high=high)
        self.wind_speed = wind_speed

    def apply_jumping_rule(self, wind_speed: float, jump_factor: float, sand_quantity: float) -> None:
        g = self.grid
        for _ in range(g.x_size * g.y_size):
            i, j = random.randint(0, g.x_size - 1), random.randint(0, g.y_size - 1)
            height = g[i, j]
            jump_length = int(wind_speed + jump_factor * height)
            g[i, j] -= sand_quantity
            g[i + jump_length, j] += sand_quantity

    def step(self) -> None:
        self.apply_jumping_rule(wind_speed=self.wind_speed, jump_factor=1.0, sand_quantity=0.1)
        self.apply_rolling_rule(diffusion_coef=0.1)


class DuneSimulator(SandSimulator):
    def __init__(self, grid_size: int = 100, sand_quantity: float = 1.5) -> None:
        super().__init__(grid_size, low=0.0, high=sand_quantity * 2.0)

    def apply_jumping_rule(self) -> None:
        g = self.grid
        for _ in range(int(0.2 * g.x_size * g.y_size)):
            i, j = random.randint(0, g.x_size - 1), random.randint(0, g.y_size - 1)
            if g[i, j] > 0.0:
                jump_length = int(11.2 - 8.0 * np.tanh(8.0 * g.partial_x(i, j)))
                sand_quantity = 0.5 + 0.4 * np.tanh(2.0 * g.partial_x(i, j))
                g[i, j] -= sand_quantity
                g[i + jump_length, j] += sand_quantity

    def step(self) -> None:
        self.apply_jumping_rule()
        self.apply_rolling_rule(diffusion_coef=0.3)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sand pattern/dunes simulation")
    parser.add_argument(
        "-w",
        "--wind_speed",
        type=float,
        metavar="V",
        default=3.0,
        help="wind speed. valid on sand pattern generation (default: 3.0)",
    )
    parser.add_argument("-g", "--grid_size", type=int, metavar="N", default=100, help="grid size (default: 100)")
    parser.add_argument("-i", "--num_iter", type=int, metavar="N", default=500, help="number of  iterations (default: 500)")
    parser.add_argument("-d", "--dune", action="store_true", help="apply settings for dune generation")
    parser.add_argument(
        "-q",
        "--quantity",
        type=float,
        metavar="Q",
        default=1.0,
        help="sand quantity. valid on dune generation (default: 1.0)",
    )
    parser.add_argument("-s", "--seed", type=int, metavar="N", default=None, help="random seed (default: None)")
    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.dune:
        simulator = DuneSimulator(args.grid_size, args.quantity)
        fig_title = f"Dune pattern. sand quantity: {args.quantity}"
        writer = SandMovieWriter(fig_title, z_min=0.0, z_max=14.0, interval=5)
    else:
        simulator = SandPatternSimulator(args.grid_size, args.wind_speed)
        fig_title = f"Sand pattern. wind speed: {args.wind_speed}"
        writer = SandMovieWriter(fig_title, z_min=-3.0, z_max=12.0, interval=1)

    # Simulate
    progress_bar = tqdm(range(args.num_iter))
    for _ in progress_bar:
        simulator.step()
        writer.plot(simulator.grid)
        progress_bar.set_postfix_str(simulator.calc_stats())

    # Save as movie
    print("Writing movie...")
    os.makedirs("movie", exist_ok=True)
    datetime_str = datetime.now().strftime("%y%m%d_%H%M%S")
    output_path = f"movie/sand_pattern_{datetime_str}.mp4"
    writer.save_movie(output_path)


if __name__ == "__main__":
    main()
