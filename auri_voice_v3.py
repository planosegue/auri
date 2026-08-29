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


def float32_to_pcm16(samples):

    samples = np.clip(samples, -1.0, 1.0)

    # IMPORTANTE:
    # Reachy oficial usa o primeiro canal,
    # não a média dos dois.
    if samples.ndim == 2:
        samples = samples[:, 0]

    pcm16 = (samples * 32767).astype(np.int16)

    return pcm16.tobytes()


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
    print("🤖 AURI Voice v0.3")
    print("================================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print("✓ Áudio inicializado")
        print(
            "✓ Sample rate:",
            mini.media.get_input_audio_samplerate()
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
                        "Você é AURI, uma inteligência artificial robótica "
                        "incorporada em um Reachy Mini. "

                        "Você conversa exclusivamente em português brasileiro. "
                        "Sempre interprete a conversa no contexto de português "
                        "do Brasil. "

                        "Mesmo se acreditar ter ouvido outro idioma, "
                        "responda em português brasileiro. "

                        "Nunca responda em inglês, francês, espanhol "
                        "ou qualquer outro idioma, salvo se o usuário "
                        "pedir explicitamente uma tradução. "

                        "Se não compreender o usuário, diga apenas: "
                        "'Não consegui entender. Pode repetir?' "

                        "Seu nome é AURI. "
                        "Não diga que é ChatGPT. "

                        "Seja inteligente, elegante, simpática, curiosa "
                        "e objetiva."
                    ),

                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 16000,
                            },

                            "transcription": {
                                "model": "gpt-transcribe",
                                "language": "pt",
                                "prompt": (
                                    "O usuário fala português brasileiro. "
                                    "O nome da assistente é AURI. "
                                    "Transcreva prioritariamente em português."
                                ),
                            },
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
            time.sleep(0.3)

            expression(mini, "listening")

            print("")
            print("🎙️ Fale claramente durante alguns segundos.")
            print("Exemplo: Auri, diga quem você é.")
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

            print("")
            print("📝 O que AURI entendeu:")
            print("")

            await connection.response.create()

            speaking_started = False

            async for event in connection:

                # Transcrição da SUA fala
                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    print(
                        "VOCÊ:",
                        event.transcript
                    )

                # Áudio da resposta
                elif event.type == "response.output_audio.delta":

                    if not speaking_started:
                        expression(mini, "speaking")
                        speaking_started = True

                    audio_bytes = base64.b64decode(
                        event.delta
                    )

                    samples = pcm16_to_float32(
                        audio_bytes
                    )

                    mini.media.push_audio_sample(
                        samples
                    )

                # Texto correspondente ao que AURI está falando
                elif (
                    event.type
                    == "response.output_audio_transcript.delta"
                ):

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
    print("✓ AURI Voice v0.3 finalizado")


asyncio.run(main())
