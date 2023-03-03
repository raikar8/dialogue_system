from TTS.api import TTS

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[0]
# Init TTS
tts = TTS(model_name)
# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])

#print(wav)

import simpleaudio as sa
import numpy as np

wav = np.array(wav)


audio = wav * (2**15 - 1) / np.max(np.abs(wav))
audio = audio.astype(np.int16)
play_obj = sa.play_buffer(audio, 1, 2, 16000)

play_obj.wait_done()


