import os
import time
from rich import print
from rich.progress import Progress
import numpy as np
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)
import torch
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
        device: str = None,
        config: dict = None,
    ):
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
            # use token mask
            # print(tokens_mask.shape)
            # print(tokens_mask.sum(),tokens_encoded[tokens_mask].shape)
            # raise Exception("testing")
            embeddings.append(tokens_encoded[tokens_mask].mean(dim=0).cpu().detach())
            break

        return embeddings

    def write_embedding(self, embedding: torch.Tensor, embedding_path: str) -> bool:
        print(f"\tsaving to '{embedding_path}'")
        try:
            torch.save(embedding, embedding_path)
        except RuntimeError:
            pass

        return os.path.isfile(embedding_path)


def main():
    supported_extensions = (".mid", ".midi")
    encoder = "data/note_encoder.bin"
    device = "cuda:1"
    in_path = "/media/nova/Datasets/sageev-midi/20250110/segmented"
    out_path = "data/embeddings"
    log_path = "data/log.txt"

    os.makedirs(out_path, exist_ok=True)

    print(f"initializing embedding generator")
    generator = EmbeddingGenerator(encoder, device)
    print(f"initialization complete")

    n_files = 0
    for path, _, files in os.walk(in_path):
        n_files += len([f for f in files if f.endswith(supported_extensions)])

    p = Progress()
    task1 = p.add_task(f"generating tokens", total=n_files)
    task2 = p.add_task(f"generating embeddings", total=n_files)
    successful_writes = 0
    with p:
        midi_tokens = dict()
        for path, _, files in os.walk(in_path):
            for file in [
                os.path.join(path, f) for f in files if f.endswith(supported_extensions)
            ]:
                out_file_path = os.path.join(
                    out_path, os.path.splitext(os.path.basename(file))[0] + ".pt"
                )
                midi_tokens[out_file_path, file] = generator.process(file)
                p.advance(task1)
            #     if len(midi_tokens)==100:
            #         break
            # if len(midi_tokens)==100:
            #     break
        keys = list(midi_tokens.keys())
        all_tokens = torch.cat([torch.IntTensor(midi_tokens[key][0]).view(1, -1) for key in keys])
        print(all_tokens.shape)
        BSZ = 8
        for i in range(0,all_tokens.shape[0],BSZ):
            batch = all_tokens[i:i+BSZ].cuda(device=generator.device)
            with torch.autocast("cuda"):
                tokens, tokens_mask = generator.get_embeddings_tokenized(batch)
            for idx in range(batch.shape[0]):
                out_file_path, file = keys[i+idx]
                avg_embedding = tokens[idx][tokens_mask[idx]].mean(0)
                
                print(f"generating embedding for '{file}'")
                print(
                    f"\tshape: {avg_embedding.shape}, min: {avg_embedding.min()}, max: {avg_embedding.max()}, mean: {avg_embedding.mean()}"
                )

                # save average embedding tensor to file, retrying once in case of corruption error
                if not generator.write_embedding(avg_embedding, out_file_path):
                    print(f"\tRuntimeError while writing file, retrying...")
                    time.sleep(1)
                    midi_embeddings = generator.get_embeddings(file)
                    avg_embedding = midi_embeddings[0] #np.mean(midi_embeddings, axis=0)
                    print(
                        f"\tshape: {avg_embedding.shape}, min: {avg_embedding.min()}, max: {avg_embedding.max()}, mean: {avg_embedding.mean()}"
                    )
                    if not generator.write_embedding(avg_embedding, out_file_path):
                        print(
                            f"\tSecond RuntimeError while writing file, logging and skipping..."
                        )
                        with open(log_path, "a") as f:
                            f.write(f"ERROR, {file}, {out_file_path},\n")

                if os.path.isfile(out_file_path):
                    successful_writes += 1
                    print(
                        f"\t{os.path.getsize(out_file_path)} bytes written to {out_file_path}"
                    )

                p.advance(task2)

    print(f"embedding generation complete")
    print(
        f"{successful_writes} / {n_files} files written, see '{log_path}' for details"
    )
    print(f"DONE")


if __name__ == "__main__":
    main()
