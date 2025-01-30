import os
import time
import h5py
import torch
import pandas as pd
from rich import print
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)
from diffusers import MidiProcessor

torch.set_grad_enabled(False)


class EmbeddingGenerator:
    encoder_config = {
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

    def __init__(
        self,
        encoder_weights: str,
        out_path: str,
        device: str = None,
        config: dict = None,
    ):
        # init outfile
        self.output_file_path = out_path
        # self.data_file = h5py.

        # build processor
        self.processor = MidiProcessor()
        if device:
            self.device = device
        else:
            self.device = "cpu"

        # build encoder
        if config:  # option to override local config
            for k, v in config.items():
                self.encoder_config[k] = v
        self.midi_encoder = SpectrogramNotesEncoder(**self.encoder_config).cuda(
            device=self.device
        )
        self.midi_encoder.eval()
        sd = torch.load(encoder_weights, weights_only=True)
        self.midi_encoder.load_state_dict(sd)

    def process(self, file_path: str) -> list[torch.tensor]:
        return self.processor(file_path)

    def get_embeddings_tokenized(self, input_tokens):
        tokens_mask = input_tokens > 0
        tokens_encoded, tokens_mask = self.midi_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
        )
        return tokens_encoded, tokens_mask

    def get_embeddings(self, file_path: str) -> list[torch.tensor]:
        out = self.process(file_path)
        embeddings = []
        for input_tokens in out:
            input_tokens = (
                torch.IntTensor(input_tokens).view(1, -1).cuda(device=self.device)
            )
            tokens_mask = input_tokens > 0
            tokens_encoded, tokens_mask = self.midi_encoder(
                encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
            )
            embeddings.append(tokens_encoded[tokens_mask].mean(dim=0).cpu().detach())
            break  # NOTE: we're only using the first 5.12 seconds!

        return embeddings

    def save_embeddings(self, results):
        t_file_start = time.perf_counter()
        with pd.HDFStore(self.output_file) as store:
            store.append("embeddings", results)
        t_file_end = time.perf_counter()
        print(f"file write took {t_file_end - t_file_start:.05f} seconds")

    def run(self, file_list: list[str], batch_size: int = 32):
        for i in range(0, len(file_list), batch_size):
            # tokenization time start
            t_tok_start = time.perf_counter()

            # generate tokens
            midi_tokens = dict()
            for idx in range(batch_size):
                if i + idx > len(file_list):
                    print(f"skipping index {i + idx} > {len(file_list)}")
                    continue
                midi_tokens[
                    os.path.splitext(os.path.basename(file_list[i + idx]))[0]
                ] = self.process(file_list[i + idx])

            batch_keys = [
                os.path.splitext(os.path.basename(k))[0] for k in midi_tokens.keys()
            ]
            # print(f"keys:\n{batch_keys}")
            batch_tokens = torch.cat(
                [torch.IntTensor(midi_tokens[key][0]).view(1, -1) for key in batch_keys]
            ).cuda(device=self.device)

            # tokenization time end & embedding time start
            t_tok_end = time.perf_counter()
            print(f"tokenization took {t_tok_end - t_tok_start:.05f} seconds")

            # generate embeddings
            series = pd.DataFrame(
                index=batch_keys, columns=range(self.encoder_config["d_model"])
            )
            with torch.autocast("cuda"):
                tokens, tokens_mask = self.get_embeddings_tokenized(batch_tokens)
            for idx in range(batch_tokens.shape[0]):
                ae = tokens[idx][tokens_mask[idx]].mean(0).cpu().detach()
                print(f"saving to {batch_keys[idx]}:\n{ae}")
                series.loc[batch_keys[idx]] = (
                    tokens[idx][tokens_mask[idx]].mean(0).cpu().detach()
                )
            self.save_embeddings(series)

            # embedding time end
            t_emb_end = time.perf_counter()
            print(f"embedding took {t_emb_end - t_tok_end:.05f} seconds")


def main():
    supported_extensions = (".mid", ".midi")
    encoder = "data/note_encoder.bin"
    device = "cuda:1"
    dataset_name = "20250110-segmented"
    in_path = "/media/nova/Datasets/sageev-midi/20250110/segmented"
    out_path = f"data/{dataset_name}.h5"
    batch_size = 4

    print(f"initializing embedding generator")
    generator = EmbeddingGenerator(encoder, out_path, device=device)
    print(f"initialization complete")

    n_files = 0
    all_files = []
    for path, _, files in os.walk(in_path):
        valid_files = [
            os.path.join(path, f) for f in files if f.endswith(supported_extensions)
        ]
        n_files += len(valid_files)
        all_files.extend(valid_files)
    all_files.sort()
    print(f"processing {n_files} files, e.g.:\n{all_files[:5]}")

    generator.run(all_files, batch_size=batch_size)

    if os.path.exists(out_path):
        print(f"saved embeddings to {out_path}")
    else:
        print(f"error saving embeddings!")

    print(f"embedding generation complete")
    print(f"wrote {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
