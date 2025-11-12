import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import imageio.v3 as iio
from rich.progress import track

from visionsim.emulate.spc import spc_avg_to_rgb
from visionsim.utils.color import linearrgb_to_srgb


def process_file(
    binary_path: Path,
    testset: Path,
    tonemap: bool,
    invert_response: bool,
    batch_size: int,
    output_path: Path,
):
    try:
        pc = np.load(binary_path)
        pc = np.unpackbits(pc[-batch_size:], axis=2)
        im = pc.sum(axis=0) / batch_size

        if invert_response:
            im = spc_avg_to_rgb(im, factor=0.5)
        if tonemap:
            im = linearrgb_to_srgb(im)

        im = (im * 255).astype(np.uint8)

        relative_path = binary_path.with_suffix(".png").relative_to(testset)
        save_path = output_path / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        iio.imwrite(save_path, im)
        return f"Saved: {save_path}"
    except Exception as e:
        return f"Failed {binary_path}: {e}"


def naivesum_parallel(
    testset: Path,
    tonemap: bool = False,
    invert_response: bool = False,
    batch_size: int = 1024,
    output_path: Path = Path("./out"),
    num_workers: int = 8,
):
    output_path.mkdir(parents=True, exist_ok=True)
    files = list(testset.glob("**/*.npy"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                f,
                testset,
                tonemap,
                invert_response,
                batch_size,
                output_path,
            )
            for f in files
        ]

        for result in track(futures, description="Processing", total=len(futures)):
            print(result.result())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process files from a directory with a given batch size."
    )

    parser.add_argument("directory", type=Path, help="Path to the input directory.")

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1024,
        help="Batch size for processing (default: 1024).",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Apply invert response filter",
    )

    parser.add_argument(
        "--tonemap", "-t", type=bool, default=True, help="Enable tonemapping"
    )

    parser.add_argument(
        "--invert-response",
        "-i",
        type=bool,
        default=True,
        help="Apply invert response filter",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./out"),
        help="Output zip file path & name",
    )

    args = parser.parse_args()

    # Validate directory
    if not args.directory.exists() or not args.directory.is_dir():
        parser.error(f"The path '{args.directory}' is not a valid directory.")

    print(f"Directory: {args.directory}")
    print(f"Batch size: {args.batch_size}")
    print(f"Invert response: {args.invert_response}")
    print(f"Tonemap: {args.tonemap}")
    print(f"Output: {args.output}")

    naivesum_parallel(
        args.directory,
        tonemap=args.tonemap,
        invert_response=args.invert_response,
        batch_size=args.batch_size,
        output_path=args.output,
        num_workers=args.workers,
    )
