import sys

# Add the path to your local TTS repository
repo_path = "/data/ericschlanger/TTS"
sys.path.insert(0, repo_path)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio.numpy_transforms import save_wav

import torch
import numpy as np

VOCAB_PATH = "/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
config = XttsConfig()
# config.load_json("/share/home/ericschlanger/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
config.load_json("/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-August-19-2024_06+09PM-dbf1a08a/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path="/data/ericschlanger/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-August-19-2024_06+09PM-dbf1a08a/best_model_8646.pth",
    vocab_path=VOCAB_PATH,
    eval=True,
)
# model.load_checkpoint(config, checkpoint_dir="/share/home/ericschlanger/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", eval=True)
model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/data/ericschlanger/TTS/example.wav",
    gpt_cond_len=3,
    language="en",
)

wav = outputs["wav"]

# if tensor convert to numpy
if torch.is_tensor(wav):
    wav = wav.cpu().numpy()
if isinstance(wav, list):
    wav = np.array(wav)
save_wav(wav=wav, path="/data/ericschlanger/TTS/from_inference9.wav", sample_rate=22050)