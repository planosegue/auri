import time
import wave

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def expression(mini, name):
    expressions = {
        "neutral": {
            "head": create_head_pose(),
            "antennas": [0.0, 0.0],
            "duration": 0.7,
        },

        "listening": {
            "head": create_head_pose(
                pitch=3,
                degrees=True,
            ),
            "antennas": [0.30, -0.30],
            "duration": 0.5,
        },

        "thinking": {
            "head": create_head_pose(
                yaw=-8,
                roll=8,
                degrees=True,
            ),
            "antennas": [0.10, -0.10],
            "duration": 0.7,
        },

        "speaking": {
            "head": create_head_pose(
                pitch=-3,
                degrees=True,
            ),
            "antennas": [0.35, -0.35],
            "duration": 0.4,
        },
    }

    state = expressions[name]

    print(f"🤖 AURI → {name}")

    mini.goto_target(
        head=state["head"],
        antennas=state["antennas"],
        duration=state["duration"],
    )


def record_audio(mini, duration=5):
    print(f"🎙️ AURI ouvindo por {duration} segundos...")

    mini.media.start_recording()

    samplerate = mini.media.get_input_audio_samplerate()

    chunks = []

    start = time.time()

    while time.time() - start < duration:
        sample = mini.media.get_audio_sample()

        if sample is not None:
            chunks.append(sample)

        time.sleep(0.01)

    mini.media.stop_recording()

    if not chunks:
        print("❌ Nenhum áudio recebido.")
        return None

    audio = np.concatenate(chunks, axis=0)

    filename = "/tmp/auri_input.wav"

    # converte float32 [-1, 1] para PCM16
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())

    print(f"✓ Áudio salvo em {filename}")

    return filename


def play_audio(mini, filename):
    print("🔊 AURI reproduzindo o que ouviu...")

    mini.media.play_sound(filename)

    # tempo suficiente para o teste
    time.sleep(6)


print("")
print("================================")
print("🤖 AURI Runtime v0.1")
print("================================")
print("")

with ReachyMini(
    media_backend="default",
) as mini:

    print("✓ Reachy Mini conectado")

    expression(mini, "neutral")
    time.sleep(0.5)

    expression(mini, "listening")

    audio_file = record_audio(
        mini,
        duration=5,
    )

    expression(mini, "thinking")
    time.sleep(1.5)

    if audio_file:

        expression(mini, "speaking")

        play_audio(
            mini,
            audio_file,
        )

    expression(mini, "neutral")

print("")
print("✓ AURI Runtime finalizado")
