import os
import h5py
import argparse
from rich import print
from rich.progress import Progress, MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn
import h5py
import torch
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)

BATCH_SIZE = 1
SUPPORTED_EXTENSIONS = (".mid", ".midi")
ENCODER_CONFIG = {
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


def embed(config):
    # initialize encoder
    midi_encoder = SpectrogramNotesEncoder(**ENCODER_CONFIG).cuda(device=config.device)
    midi_encoder.eval()
    sd = torch.load("data/note_encoder.bin", weights_only=True)
    midi_encoder.load_state_dict(sd)

    with h5py.File(config.in_file, "r") as in_file:
        # load input datasets
        files = in_file["filenames"]
        print(f"processing {len(files)} files, e.g.:\n{files[:5]}")
        tokens = in_file["tokens"]
        print(f"processing {len(tokens)} files, e.g.:\n{tokens[:5]}")

        with h5py.File(config.out_file, "a") as out_file:
            # create output datasets
            d_embeddings = out_file.create_dataset(
                "embeddings", (len(files), ENCODER_CONFIG["d_model"]), fillvalue=0
            )
            d_filenames = out_file.create_dataset(
                "filenames",
                (len(files), 1),
                dtype=h5py.string_dtype(encoding="utf-8"),
                fillvalue="",
            )

            # calculate embeddings while tracking progress
            progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                refresh_per_second=1,
            )
            tok_task = progress.add_task("embedding", total=len(files))
            with progress:
                for i in range(0, len(files), BATCH_SIZE):
                    with torch.autocast("cuda"):
                        batch_tokens = torch.cat(
                            [torch.IntTensor(tokens[i : i + BATCH_SIZE])]
                        ).cuda(device=config.device)
                        # print(f"loaded tokens {batch_tokens.shape}")
                        tokens_mask = batch_tokens > 0
                        tokens_encoded, tokens_mask = midi_encoder(
                            encoder_input_tokens=batch_tokens,
                            encoder_inputs_mask=tokens_mask,
                        )
                    # print(f"generated embeddings {tokens_encoded.shape}")
                    # TODO: normalize each embedding before storing it
                    d_embeddings[i : i + BATCH_SIZE] = [
                        enc[mask].mean(0).cpu().detach()
                        for enc, mask in zip(tokens_encoded, tokens_mask)
                    ]
                    # print(f"stored embeddings", d_embeddings[i : i + BATCH_SIZE])
                    d_filenames[i : i + BATCH_SIZE] = files[i : i + BATCH_SIZE]
                    # print(f"stored filenames", d_filenames[i : i + BATCH_SIZE])
                    progress.advance(tok_task, BATCH_SIZE)

    # verify outputs
    with h5py.File(config.out_file, "r") as f:
        print(f"stored tokens in HDF5 file with filenames:")
        for filename in f["filenames"][:5]:
            print(f"\t{str(filename[0], 'utf-8')}")
        print(f"and embeddings ({f['embeddings'].shape}):")
        for embeddings in f["embeddings"][:5]:
            print(f"\t{embeddings}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Spectrogram Diffusion Embeddings", add_help=False)
    parser.add_argument("--in_file", "-i", type=str)
    parser.add_argument("--out_file", "-o", type=str)
    parser.add_argument("--device", "-d", type=str, default="cuda:1")
    config = parser.parse_args()

    embed(config)
