import asyncio
import base64
import os
import time

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


MODEL = "gpt-realtime-2.1-mini"


def expression(mini, name):
    states = {
        "neutral": {
            "head": create_head_pose(),
            "antennas": [0.0, 0.0],
            "duration": 0.6,
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


def float32_to_pcm16(samples):
    samples = np.clip(samples, -1.0, 1.0)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return (samples * 32767).astype(np.int16).tobytes()


def pcm16_to_float32(data):
    pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


async def main():

    load_dotenv("/home/pollen/auri/.env")

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("")
    print("================================")
    print("🤖 AURI Voice v0.2")
    print("================================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print("✓ Áudio inicializado")

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
                        "Seu nome é AURI. "
                        "Você é uma inteligência artificial incorporada "
                        "em um robô Reachy Mini. "

                        "REGRA ABSOLUTA: "
                        "FALE SEMPRE E EXCLUSIVAMENTE EM PORTUGUÊS DO BRASIL, "
                        "independentemente do idioma, ruído, som ou palavra "
                        "que você acreditar ter ouvido. "
                        "Nunca responda em inglês. "

                        "Se o áudio estiver incompreensível, diga em português: "
                        "'Não consegui entender muito bem. Pode repetir?' "

                        "Sua personalidade é inteligente, elegante, curiosa, "
                        "natural e levemente bem-humorada. "

                        "Não diga que é ChatGPT. "
                        "Você é AURI. "

                        "Responda de maneira curta e conversacional nesta fase "
                        "de testes."
                    ),

                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 16000,
                            }
                        },

                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 16000,
                            },

                            "voice": "marin",
                        },
                    },
                }
            )

            expression(mini, "neutral")
            time.sleep(0.4)

            expression(mini, "listening")

            print("")
            print("🎙️ Fale com AURI por 5 segundos...")
            print("")

            chunks_sent = 0

            loop = asyncio.get_running_loop()
            end_time = loop.time() + 5.0

            while loop.time() < end_time:

                samples = mini.media.get_audio_sample()

                if samples is None:
                    await asyncio.sleep(0.01)
                    continue

                pcm = float32_to_pcm16(samples)

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

            print("🧠 AURI pensando...")

            await connection.response.create()

            transcript = ""

            async for event in connection:

                if event.type == "response.output_audio.delta":

                    if transcript == "":
                        expression(mini, "speaking")

                    audio_bytes = base64.b64decode(
                        event.delta
                    )

                    samples = pcm16_to_float32(
                        audio_bytes
                    )

                    mini.media.push_audio_sample(
                        samples
                    )

                elif event.type == "response.output_audio_transcript.delta":

                    transcript += event.delta

                    print(
                        event.delta,
                        end="",
                        flush=True
                    )

                elif event.type == "response.done":

                    print("")
                    print("")
                    print("✓ Resposta finalizada")

                    break

            await asyncio.sleep(0.5)

            expression(mini, "neutral")

        mini.media.stop_recording()
        mini.media.stop_playing()

    print("")
    print("✓ AURI Voice v0.2 finalizado")


asyncio.run(main())
