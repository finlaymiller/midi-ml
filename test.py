from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import SpectrogramNotesEncoder
import torch 
from diffusers import MidiProcessor 

torch.set_grad_enabled(False)

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
  "vocab_size": 1536
}

processor = MidiProcessor()
notes_encoder = SpectrogramNotesEncoder(**cfg).cuda()
notes_encoder.eval() 

sd = torch.load("data/note_encoder.bin")
print(notes_encoder.load_state_dict(sd))

out = processor("data/beethoven_hammerklavier_2.mid")
print(len(out))
print(torch.IntTensor(out).shape)
for input_tokens in out:
    
    # break
    input_tokens = torch.IntTensor(input_tokens).view(1,-1).cuda()
    print(input_tokens[:,:20])
    tokens_mask = input_tokens > 0
    cutoff = (input_tokens > 0).sum()
    print(tokens_mask.sum())
    tokens_encoded, tokens_mask = notes_encoder(
        encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
    )
    print(tokens_encoded.shape, tokens_encoded[0,cutoff:].sum())# (tokens_encoded**2).sum())
