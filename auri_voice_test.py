import asyncio
import base64
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from reachy_mini import ReachyMini


MODEL = "gpt-realtime-2.1-mini"


def float32_to_pcm16(samples):
    # Reachy entrega float32 [-1,1]
    samples = np.clip(samples, -1.0, 1.0)

    # estéreo -> mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    pcm16 = (samples * 32767).astype(np.int16)

    return pcm16.tobytes()


def pcm16_to_float32(data):
    pcm = np.frombuffer(data, dtype=np.int16)

    samples = pcm.astype(np.float32) / 32768.0

    return samples


async def main():

    load_dotenv("/home/pollen/auri/.env")

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("")
    print("==============================")
    print("🤖 AURI Voice Test")
    print("==============================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print("✓ Microfone e speaker iniciados")

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
                        "Fale sempre em português do Brasil. "
                        "Seja natural, inteligente, simpática e objetiva. "
                        "Nesta demonstração responda de forma curta."
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

            print("")
            print("🎙️ Fale com AURI por 5 segundos...")
            print("")

            # grava aproximadamente 5 segundos
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

            await connection.response.create()

            print("🧠 AURI pensando...")
            print("🔊 Resposta:")

            async for event in connection:

                if event.type == "response.output_audio.delta":

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

                    print(
                        event.delta,
                        end="",
                        flush=True
                    )

                elif event.type == "response.done":

                    print("")
                    print("✓ Resposta finalizada")
                    break

        mini.media.stop_recording()
        mini.media.stop_playing()

    print("")
    print("✓ AURI Voice Test finalizado")


asyncio.run(main())
