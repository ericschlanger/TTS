import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
CONFIG_PATH = "/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-August-19-2024_12+08PM-dbf1a08a/config.json"
# Add here the vocab file that you have used to train the model
VOCAB_PATH = "/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
# XTTS_CHECKPOINT = "/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/model.pth"
XTTS_CHECKPOINT = "/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-August-28-2024_11+32AM-496e5a8c/checkpoint_3365.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "/data/ericschlanger/TTS/example.wav"

# output wav path
OUTPUT_WAV_PATH = "xtts-ft3.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=XTTS_CHECKPOINT,
    vocab_path=VOCAB_PATH,
    use_deepspeed=False,
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)