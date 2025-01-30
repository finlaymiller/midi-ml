import os
from rich import print
from rich.progress import Progress
from diffusers import MidiProcessor
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    labels = []
    device = "cuda:1"
    supported_extensions = (".mid", ".midi")
    cfg = {
        #   "_class_name": "SpectrogramNotesEncoder",
        #   "_diffusers_version": "0.14.0.dev0",
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "dropout_rate": 0.1,
        "feed_forward_proj": "gated-gelu_pytorch_tanh",
        "is_decoder": False,
        "max_length": 2048,
        "num_heads": 12,
        "num_layers": 12,
        "vocab_size": 1536,
    }

    def __init__(self, data_dir: str, transform=None, target_transform=None, cfg=None):
        # read in parameters
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        if cfg:  # option to override local config
            for k, v in cfg.items():
                self.cfg[k] = v

        # load processor
        self.processor = MidiProcessor()

        # scan directory for files
        num_files = 0
        for folder, _, files in os.walk(self.data_dir):
            num_files += len(files)

        p = Progress()
        tok_task = p.add_task(f"tokenizing", total=num_files)
        with p:
            for folder, _, files in os.walk(self.data_dir):
                print(f"tokenizing segments from track '{folder}'")
                for file in files:
                    if file.endswith(self.supported_extensions):
                        midi_path = os.path.join(folder, file)
                        file_name = os.path.splitext(file)[0]
                        tokens = self.processor(midi_path)
                        # print( f"got {len(tokens):03d} tokens for '{midi_path}'", end="\r")
                        self.labels.append((midi_path, file_name, tokens))
                        p.advance(tok_task)

        print(f"loaded {len(self.labels)} MIDI files")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        midi_path, label, tokens = self.labels[idx]

        if self.transform:
            midi_path = self.transform(midi_path)
        if self.target_transform:
            label = self.target_transform(label)

        return midi_path, label, tokens

    def check_len(self, idx: int):
        import pretty_midi

        midi_path, label, tokens = self.labels[idx]
        midi = pretty_midi.PrettyMIDI(midi_path)
        print(
            f"'{label}' is {midi.get_end_time():.03f} seconds and generated {len(tokens)} tokens"
        )
        print(f"{len(tokens)} * 5s = {len(tokens) * 5}s")

        if abs(midi.get_end_time() - (len(tokens) * 5)) < 5:
            print(f"number of tokens is close enough to MIDI file length")
        else:
            print(f"[red]something is wrong with the number of tokens generated")


if __name__ == "__main__":
    test_path = "/media/nova/Datasets/sageev-midi/20250110/unsegmented"
    test_index = 0
    print(
        f"Testing custom dataset class on default path '{test_path}' at index {test_index}"
    )

    dataset = MidiDataset(test_path)
    m, l, t = dataset[test_index]
    print(
        f"element 0 in dataset '{l}' with path '{m}' has ({len(t)}, {len(t[0])}) tokens"
    )
    dataset.check_len(test_index)
    print(f"[green bold]TEST COMPLETE")
