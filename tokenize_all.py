import os
import h5py
import torch
import argparse
from rich import print
from pretty_midi import PrettyMIDI
from diffusers import MidiProcessor
from mido import MidiFile, bpm2tempo, MetaMessage, MidiTrack
from rich.progress import Progress, MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn

SUPPORTED_EXTENSIONS = (".mid", ".midi")

def change_tempo_and_trim(
    input_file: str, output_file: str, tempo: float = 103.47, cutoff_sec: float = 5.12
) -> bool:
    """Process a MIDI file to change its tempo and trim events after a cutoff.

    This function loads a MIDI file, replaces any existing tempo messages with a new tempo
    message, and trims events so that no event occurs after the specified cutoff time.
    If any note remains active at the cutoff, a note_off message is inserted.

    Args:
        input_file (str): Path to the input MIDI file.
        output_file (str): Path where the processed MIDI file will be saved.
        tempo (float, optional): Desired tempo in BPM. Defaults to 103.47 so that 9 beats equals 5.12s.
        cutoff_sec (float, optional): Time in seconds after which events are trimmed.
            Defaults to 5.12.

    Returns:
        bool: True if the output file exists after saving, False otherwise.
    """
    import os
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage

    mid = MidiFile(input_file)
    new_tempo_value = mido.bpm2tempo(tempo)
    ticks_per_beat = mid.ticks_per_beat

    # if no tracks exist, add a new track with the tempo message
    if not mid.tracks:
        new_track = MidiTrack()
        new_track.append(MetaMessage("set_tempo", tempo=new_tempo_value, time=0))
        mid.tracks.append(new_track)

    new_tracks = []
    for track_index, track in enumerate(mid.tracks):
        new_track = []
        current_abs_tick = 0
        active_notes = {}
        # insert new tempo message at beginning of first track
        if track_index == 0:
            new_track.append(MetaMessage("set_tempo", tempo=new_tempo_value, time=0))
        for msg in track:
            # skip existing tempo messages
            if msg.type == "set_tempo":
                continue
            next_abs_tick = current_abs_tick + msg.time
            event_time_sec = mido.tick2second(
                next_abs_tick, ticks_per_beat, new_tempo_value
            )
            if event_time_sec > cutoff_sec:
                current_time_sec = mido.tick2second(
                    current_abs_tick, ticks_per_beat, new_tempo_value
                )
                remaining_sec = cutoff_sec - current_time_sec
                remaining_ticks = int(
                    mido.second2tick(remaining_sec, ticks_per_beat, new_tempo_value)
                )
                first = True
                for channel, note in list(active_notes.keys()):
                    note_off_time = remaining_ticks if first else 0
                    new_track.append(
                        mido.Message(
                            "note_off",
                            note=note,
                            channel=channel,
                            velocity=0,
                            time=note_off_time,
                        )
                    )
                    first = False
                current_abs_tick += remaining_ticks
                break
            else:
                new_track.append(msg.copy(time=msg.time))
                current_abs_tick = next_abs_tick
                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[(msg.channel, msg.note)] = True
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        del active_notes[key]
        current_time_sec = mido.tick2second(
            current_abs_tick, ticks_per_beat, new_tempo_value
        )
        if current_time_sec < cutoff_sec and active_notes:
            remaining_sec = cutoff_sec - current_time_sec
            remaining_ticks = int(
                mido.second2tick(remaining_sec, ticks_per_beat, new_tempo_value)
            )
            first = True
            for channel, note in list(active_notes.keys()):
                note_off_time = remaining_ticks if first else 0
                new_track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        channel=channel,
                        velocity=0,
                        time=note_off_time,
                    )
                )
                first = False
        new_tracks.append(new_track)

    mid.tracks = new_tracks
    mid.save(output_file)
    return os.path.isfile(output_file)


def tokenize(config):
    processor = MidiProcessor()

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
    if config.test:
        import random
        import numpy as np

        random.seed(0)
        n_files = 100
        all_files = random.sample(all_files, n_files)
    print(f"processing {n_files} files, e.g.:\n{all_files[:5]}")

    with h5py.File(config.out_file, "a") as f:
        # create datasets
        d_tokens = f.create_dataset("tokens", (n_files, 2048), fillvalue=0)
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
        tok_task = progress.add_task("tokenizing", total=n_files)
        with progress:
            for i, file in enumerate(all_files):
                if config.fix_tempo:
                    # print(MidiFile(file).print_tracks())
                    change_tempo_and_trim(file, file)
                    # print(MidiFile(file).print_tracks())
                    # print(
                    #     f"new length of '{file}' is {PrettyMIDI(file).get_end_time()}"
                    # )

                tokens = processor(file)
                if config.test:
                    print(
                        f"generated {len(tokens)} tokens ({np.count_nonzero(torch.IntTensor(tokens[0]).view(1, -1))} nonzero)"
                    )
                d_tokens[i] = processor(file)
                d_filenames[i] = os.path.splitext(os.path.basename(file))[0] + "_t00s00"
                progress.advance(tok_task)

    # verify outputs
    with h5py.File(config.out_file, "r") as f:
        print(f"stored tokens in HDF5 file with filenames:")
        for filename in f["filenames"][:5]:
            print(f"\t{str(filename[0], 'utf-8')}")
        print(f"and tokens ({f['tokens'].shape}):")
        for tokens in f["tokens"][:5]:
            print(f"\t{tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Spectrogram Diffusion Tokens", add_help=False)
    parser.add_argument("--in_path", "-i", type=str)
    parser.add_argument("--out_file", "-o", type=str)
    parser.add_argument("--fix_tempo", "-f", action="store_true", default=False)
    parser.add_argument("--test", "-t", action="store_true", default=False)
    config = parser.parse_args()
    print(config)

    if config.test and os.path.isfile(config.out_file):
        print(f"deleting existing test file: '{config.out_file}'")
        os.remove(config.out_file)

    tokenize(config)
