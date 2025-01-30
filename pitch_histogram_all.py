import os
import h5py
import argparse
import pretty_midi
from rich import print
from rich.progress import Progress, MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn

SUPPORTED_EXTENSIONS = (".mid", ".midi")


def go(config):

    # gather file list
    n_files = 0
    all_files = []
    for path, _, files in os.walk(config.in_path):
        valid_files = [
            os.path.join(path, f) for f in files if f.endswith(SUPPORTED_EXTENSIONS)
        ]
        n_files += len(valid_files)
        all_files.extend(valid_files)
    all_files.sort()
    print(f"processing {n_files} files, e.g.:\n{all_files[:5]}")

    with h5py.File(config.out_file, "a") as f:
        # create datasets
        d_tokens = f.create_dataset("histograms", (n_files, 12), fillvalue=0)
        d_filenames = f.create_dataset(
            "filenames",
            (n_files, 1),
            dtype=h5py.string_dtype(encoding="utf-8"),
            fillvalue="",
        )

        # tokenize while tracking progress
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=1,
        )
        process_task = progress.add_task("processing", total=n_files)
        with progress:
            for i, file in enumerate(all_files):
                d_tokens[i] = pretty_midi.PrettyMIDI(file).get_pitch_class_histogram(
                    True, True
                )
                d_filenames[i] = os.path.splitext(os.path.basename(file))[0]
                progress.advance(process_task)

    # verify outputs
    with h5py.File(config.out_file, "r") as f:
        print(f"stored tokens in HDF5 file with filenames:")
        for filename in f["filenames"][:5]:
            print(f"\t{str(filename[0], 'utf-8')}")
        print(f"and histograms ({f['histograms'].shape}):")
        for histogram in f["histograms"][:5]:
            print(f"\t{histogram}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Spectrogram Diffusion Tokens", add_help=False)
    parser.add_argument("--in_path", "-i", type=str)
    parser.add_argument("--out_file", "-o", type=str)
    config = parser.parse_args()

    go(config)
