import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional
import argparse


# COLORS = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

def read_density_bin(path: str):
    with open(path, "rb") as f:
        N = np.fromfile(f, np.int32, 1)[0]
        T = np.fromfile(f, np.int32, 1)[0]
        sz = np.fromfile(f, np.int32, 1)[0]
        data = np.fromfile(f, np.float32, sz * T)
    data = data.reshape(T, N + 2, N + 2)
    frames = data[:, 1:N+1, 1:N+1]
    return N, T, frames

def render_video(frames: np.ndarray,out_path: str,fps: int = 60,vmax: Optional[float] = None,cmap: str = "inferno"):
    T, H, W = frames.shape

    if vmax is None:
        vmax = float(np.percentile(frames, 99.5))
        if vmax <= 0.0:
            vmax = 1.0

    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(
        frames[0],
        origin="lower",
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",  # smoother display when frames are scaled
        animated=True,
    )

    def update(t):
        im.set_array(frames[t])
        return (im,)

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)

    ani.save(out_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render fluid density frames to video")
    parser.add_argument("--input", default="density.bin", help="Input density binary path")
    parser.add_argument("--out", default="out.gif", help="Output path")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Color palette"
    )
    parser.add_argument("--vmax", type=float, default=None, help="Manual max density value")
    args = parser.parse_args()

    print("\nRendering density video...")
    N, T, frames = read_density_bin(f"build/{args.input}")
    print(f"N={N}, frames={T}, shape={frames.shape}")

    render_video(frames, f"sims/{args.out}", fps=args.fps, vmax=args.vmax, cmap=args.cmap)

    print(f"Video complete! ({args.out})")