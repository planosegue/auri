import asyncio
import base64
import os
import time

import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import AsyncOpenAI

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


MODEL = "gpt-realtime-2.1-mini"

REACHY_RATE = 16000
OPENAI_RATE = 24000


def expression(mini, name):
    states = {
        "neutral": {
            "head": create_head_pose(),
            "antennas": [0.0, 0.0],
            "duration": 0.6,
        },
        "listening": {
            "head": create_head_pose(pitch=3, degrees=True),
            "antennas": [0.30, -0.30],
            "duration": 0.5,
        },
        "thinking": {
            "head": create_head_pose(yaw=-8, roll=8, degrees=True),
            "antennas": [0.10, -0.10],
            "duration": 0.7,
        },
        "speaking": {
            "head": create_head_pose(pitch=-3, degrees=True),
            "antennas": [0.38, -0.38],
            "duration": 0.4,
        },
    }

    state = states[name]

    print(f"🤖 AURI → {name}")

    mini.goto_target(
        head=state["head"],
        antennas=state["antennas"],
        duration=state["duration"],
    )


def reachy_to_openai(samples):

    # Reachy: float32, stereo, 16 kHz
    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(samples, -1.0, 1.0)

    # 16 kHz -> 24 kHz
    samples_24k = resample_poly(
        samples,
        3,
        2
    )

    pcm16 = (
        np.clip(samples_24k, -1.0, 1.0)
        * 32767
    ).astype(np.int16)

    return pcm16.tobytes()


def openai_to_reachy(data):

    # OpenAI: PCM16 mono 24 kHz
    pcm = np.frombuffer(
        data,
        dtype=np.int16
    )

    samples_24k = (
        pcm.astype(np.float32)
        / 32768.0
    )

    # 24 kHz -> 16 kHz
    samples_16k = resample_poly(
        samples_24k,
        2,
        3
    ).astype(np.float32)

    return samples_16k


async def main():

    load_dotenv("/home/pollen/auri/.env")

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("")
    print("================================")
    print("🤖 AURI Voice v0.4")
    print("   Audio 16k ↔ 24k")
    print("================================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print(
            "✓ Reachy input:",
            mini.media.get_input_audio_samplerate(),
            "Hz"
        )

        async with client.realtime.connect(
            model=MODEL
        ) as connection:

            print("✓ OpenAI Realtime conectado")

            await connection.session.update(
                session={
                    "type": "realtime",
                    "model": MODEL,

                    "output_modalities": [
                        "audio"
                    ],

                    "instructions": (
                        "Você é AURI, uma inteligência artificial "
                        "incorporada em um robô Reachy Mini. "
                        "Converse sempre em português brasileiro. "
                        "O usuário principal se chama Luciano. "
                        "Seja natural, inteligente, elegante, "
                        "curiosa e objetiva. "
                        "Se não compreender claramente o usuário, "
                        "pergunte em português se ele pode repetir. "
                        "Nunca invente o que acredita ter ouvido."
                    ),

                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": OPENAI_RATE,
                            },

                            "transcription": {
                                "model": "gpt-transcribe",
                                "language": "pt",
                                "prompt": (
                                    "Português brasileiro. "
                                    "O nome do usuário é Luciano. "
                                    "O nome da assistente é AURI."
                                ),
                            },
                        },

                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": OPENAI_RATE,
                            },

                            "voice": "marin",
                        },
                    },
                }
            )

            expression(mini, "listening")

            print("")
            print("🎙️ Diga claramente:")
            print("   Eu sou o Luciano.")
            print("")

            loop = asyncio.get_running_loop()
            end_time = loop.time() + 5.0

            chunks_sent = 0

            while loop.time() < end_time:

                samples = mini.media.get_audio_sample()

                if samples is None:
                    await asyncio.sleep(0.01)
                    continue

                pcm = reachy_to_openai(samples)

                encoded = base64.b64encode(
                    pcm
                ).decode("ascii")

                await connection.input_audio_buffer.append(
                    audio=encoded
                )

                chunks_sent += 1

                await asyncio.sleep(0)

            print(
                f"✓ Áudio enviado ({chunks_sent} blocos)"
            )

            await connection.input_audio_buffer.commit()

            expression(mini, "thinking")

            # Primeiro aguardamos a transcrição do usuário.
            user_transcript = None

            print("")
            print("📝 Transcrição:")

            async for event in connection:

                if event.type == \
                    "conversation.item.input_audio_transcription.completed":

                    user_transcript = event.transcript

                    print(
                        "VOCÊ:",
                        user_transcript
                    )

                    break

                elif event.type == "error":
                    print(
                        "ERRO:",
                        event
                    )
                    break

            # Só depois pedimos resposta.
            await connection.response.create()

            speaking_started = False

            print("")
            print("AURI:", end=" ", flush=True)

            async for event in connection:

                if event.type == "response.output_audio.delta":

                    if not speaking_started:
                        expression(mini, "speaking")
                        speaking_started = True

                    audio_bytes = base64.b64decode(
                        event.delta
                    )

                    samples = openai_to_reachy(
                        audio_bytes
                    )

                    mini.media.push_audio_sample(
                        samples
                    )

                elif event.type == \
                    "response.output_audio_transcript.delta":

                    print(
                        event.delta,
                        end="",
                        flush=True
                    )

                elif event.type == "response.done":

                    print("")
                    break

            await asyncio.sleep(0.7)

            expression(mini, "neutral")

        mini.media.stop_recording()
        mini.media.stop_playing()

    print("")
    print("✓ AURI Voice v0.4 finalizado")


asyncio.run(main())
