import whisperx
import os
import json
import torch

# python -m pytorch_lightning.utilities.upgrade_checkpoint ../miniconda/envs/whisperx/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def format_timestamp(seconds: float) -> str:
    """Converts time in seconds to SRT timestamp format (HH:MM:SS,MS)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def transcribe_with_speaker_diarization(
    directory: str, model_size: str = "medium", device: str = "cuda"
) -> None:
    """Transcribes all audio files in a directory using WhisperX with speaker diarization and timestamps.

    Args:
        directory (str): Path to the directory containing audio files.
        model_size (str, optional): Whisper model size to use ("tiny", "base", "small", "medium", "large"). Defaults to "medium".
        device (str, optional): "cuda" for GPU acceleration or "cpu". Defaults to "cuda".

    Saves:
        - A `.json` file with structured transcription and speaker labels.
        - A `.srt` subtitle file with timestamps.
    """
    # load whisper model
    model_dir = "model/"
    model = whisperx.load_model(
        whisper_arch=model_size,
        device=device,
        language="en",
        compute_type="float16",
        download_root=model_dir,
    )

    # create output directory
    output_dir = os.path.join(directory, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)

    # process each audio file
    for filename in os.listdir(directory):
        if filename.endswith(
            (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        ):  # supported formats
            audio_path = os.path.join(directory, filename)
            print(f"Processing: {filename}...")

            # transcribe audio with timestamps
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=8)
            print(result["segments"])  # before alignment

            # align words with timestamps
            align_model, align_metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            aligned_result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            print(aligned_result)

            # perform speaker diarization
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=True, device=device
            )
            diarize_segments = diarize_model(audio_path)

            # assign speaker labels
            result_with_speakers = whisperx.assign_word_speakers(
                diarize_segments, aligned_result
            )
            result_with_speakers = aligned_result

            # save transcription as JSON
            json_output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.json"
            )
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result_with_speakers, f, indent=4)

            print(f"Saved JSON transcription: {json_output_path}")

            # save transcription as SRT
            srt_output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.srt"
            )
            with open(srt_output_path, "w", encoding="utf-8") as f:
                for idx, word in enumerate(result_with_speakers["word_segments"]):
                    start_time = word["start"]
                    end_time = word["end"]
                    speaker = word.get("speaker", "Unknown")
                    text = word["text"]

                    f.write(f"{idx + 1}\n")
                    f.write(
                        f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
                    )
                    f.write(f"{speaker}: {text}\n\n")

            print(f"Saved SRT subtitles: {srt_output_path}")


if __name__ == "__main__":
    # specify directory containing audio files
    audio_directory = "data/talks"  # update this
    transcribe_with_speaker_diarization(
        audio_directory, model_size="large-v3", device="cuda"
    )
