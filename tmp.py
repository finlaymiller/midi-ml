from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import SpectrogramNotesEncoder
import torch
from diffusers import MidiProcessor

torch.set_grad_enabled(False)

# Configuration for SpectrogramNotesEncoder
cfg = {
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 768,
    "dropout_rate": 0.1,
    "feed_forward_proj": "gated-gelu_pytorch_tanh",
    "max_length": 2048,
    "num_heads": 12,
    "num_layers": 12,
    "vocab_size": 1536,
    "is_decoder": False,
}

# Initialize processor and encoder
processor = MidiProcessor()
notes_encoder = SpectrogramNotesEncoder(**cfg).cuda()
notes_encoder.eval()

# Load state dictionary for the encoder
sd = torch.load("data/note_encoder.bin", weights_only=True)
print("State dictionary loaded:", notes_encoder.load_state_dict(sd))

# Process MIDI file
out = processor("data/beethoven_hammerklavier_2.mid")
print("Number of token sequences:", len(out))
print("Shape of processed output:", torch.IntTensor(out).shape)

# Debugging and processing each input sequence
for input_tokens in out:
    input_tokens = torch.IntTensor(input_tokens).view(1, -1).cuda()
    print("Input tokens (first 20):", input_tokens[:, :20])

    # Create mask and cutoff
    tokens_mask = input_tokens > 0
    cutoff = (input_tokens > 0).sum()
    print("Cutoff (non-zero tokens):", cutoff.item())

    # Add debugging: Check mask and input shapes
    print("Input shape:", input_tokens.shape)
    print("Mask shape:", tokens_mask.shape)

    # Attempt encoding
    try:
        tokens_encoded, tokens_mask = notes_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
        )
        print("Encoded output shape:", tokens_encoded.shape)
        print("Sum of unused tokens:", tokens_encoded[0, cutoff:].sum().item())
    except Exception as e:
        print("Error during encoding:", e)

# Validate encoder configuration compatibility
print("Encoder configuration:", cfg)

# Debugging: Check model internals
print("Checking T5 model internals...")
for name, param in notes_encoder.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape}")
