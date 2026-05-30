import os
import json
from pathlib import Path
from datetime import datetime

import main_dynamic
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User settings
# -----------------------------

OUTPUT_DIR = Path("generated_paths")
N_CANDIDATES = 5          # Number of paths to generate in one run
SAVE_ALL = False          # Set True to save every generated path automatically
MAX_ITERATIONS = 200        # Max iterations for PSO


# -----------------------------
# Path generation
# -----------------------------

def run_main():
    """
    Runs the path generation routine from main_dynamic.py.

    Assumes final_path_details returns:
        u, path, curvature, radius, heading, Nbestsol
    """

    u, path, curvature, radius, heading, bestsol = main_dynamic.final_path_details("bestsol")

    main_dynamic.print_final_solution_details(bestsol)

    results = {
        "u": np.asarray(u),
        "path": np.asarray(path, dtype=float),
        "curvature": np.asarray(curvature, dtype=float).ravel(),
        "radius": np.asarray(radius, dtype=float).ravel(),
        "heading": np.asarray(heading, dtype=float).ravel(),
        "bestsol": bestsol,
    }

    return results


# -----------------------------
# Plotting
# -----------------------------

def plot_path(results, candidate_number=None):
    path = results["path"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(path[:, 0], path[:, 1], linewidth=1.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if candidate_number is None:
        ax.set_title("Generated Spline Path")
    else:
        ax.set_title(f"Generated Spline Path - Candidate {candidate_number}")

    plt.tight_layout()
    plt.show(block=True)

    return fig


# -----------------------------
# Saving helpers
# -----------------------------

def ensure_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_next_file_stem(prefix="path"):
    """
    Creates a unique filename stem:
        path_001, path_002, path_003, ...
    """

    ensure_output_folder()

    existing = list(OUTPUT_DIR.glob(f"{prefix}_*.npz"))
    existing_numbers = []

    for file in existing:
        try:
            number = int(file.stem.split("_")[-1])
            existing_numbers.append(number)
        except ValueError:
            pass

    next_number = max(existing_numbers, default=0) + 1
    return f"{prefix}_{next_number:03d}"


def pad_or_nan(values, target_length):
    """
    Converts a vector to length target_length.
    If it cannot be matched, returns NaNs.
    """

    values = np.asarray(values).ravel()

    if len(values) == target_length:
        return values

    return np.full(target_length, np.nan)


def save_path_csv(results, csv_path):
    """
    Saves path-related arrays in a readable CSV format.

    The CSV is mainly for inspection. The .npz file is the main
    file to reload for simulations.
    """

    path = results["path"]
    n = path.shape[0]

    curvature = pad_or_nan(results["curvature"], n)
    radius = pad_or_nan(results["radius"], n)
    heading = pad_or_nan(results["heading"], n)

    data = np.column_stack([
        path[:, 0],
        path[:, 1],
        curvature,
        radius,
        heading,
    ])

    header = "x_m,y_m,curvature_1_per_m,radius_m,heading_rad"

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def save_path_details(results, fig=None, prefix="path"):
    """
    Saves:
        .npz  full numerical data for reloading
        .csv  readable path data
        .png  preview plot
        .json small metadata file
    """

    ensure_output_folder()

    stem = get_next_file_stem(prefix)

    npz_path = OUTPUT_DIR / f"{stem}.npz"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    png_path = OUTPUT_DIR / f"{stem}.png"
    json_path = OUTPUT_DIR / f"{stem}_metadata.json"

    # Main reloadable save file.
    # bestsol is saved as an object, so loading requires allow_pickle=True.
    np.savez_compressed(
        npz_path,
        u=results["u"],
        path=results["path"],
        curvature=results["curvature"],
        radius=results["radius"],
        heading=results["heading"],
        bestsol=np.array(results["bestsol"], dtype=object),
    )

    # Human-readable CSV for quick checking.
    save_path_csv(results, csv_path)

    # Preview plot.
    if fig is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # Small metadata file.
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "npz_file": str(npz_path),
        "csv_file": str(csv_path),
        "png_file": str(png_path),
        "number_of_path_points": int(results["path"].shape[0]),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved path:")
    print(f"  NPZ:  {npz_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  PNG:  {png_path}")
    print(f"  JSON: {json_path}")

    return npz_path


# -----------------------------
# Loading helper
# -----------------------------

def load_path_details(npz_path):
    """
    Reloads a previously generated path.

    Example:
        results = load_path_details("generated_paths/path_001.npz")
        path = results["path"]
    """

    loaded = np.load(npz_path, allow_pickle=True)

    results = {
        "u": loaded["u"],
        "path": loaded["path"],
        "curvature": loaded["curvature"],
        "radius": loaded["radius"],
        "heading": loaded["heading"],
        "bestsol": loaded["bestsol"].item()
        if loaded["bestsol"].shape == ()
        else loaded["bestsol"],
    }

    return results


# -----------------------------
# Main loop
# -----------------------------

def main():
    ensure_output_folder()

    saved_paths = []

    for i in range(1, N_CANDIDATES + 1):
        print(f"\nGenerating candidate path {i}/{N_CANDIDATES}...\n")

        results = run_main()
        fig = plot_path(results, candidate_number=i)

        if SAVE_ALL:
            saved_path = save_path_details(results, fig=fig)
            saved_paths.append(saved_path)
        else:
            decision = input("Save this path? [y/N/q]: ").strip().lower()

            if decision == "q":
                print("Stopped path generation.")
                plt.close(fig)
                break

            if decision == "y":
                saved_path = save_path_details(results, fig=fig)
                saved_paths.append(saved_path)
            else:
                print("Path discarded.")

        plt.close(fig)

    print("\nFinished.")
    print(f"Saved {len(saved_paths)} path(s).")


if __name__ == "__main__":
    main()