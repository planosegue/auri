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

OPENAI_RATE = 24000

# Janela curta depois que AURI termina de falar.
# Durante esse tempo ignoramos um novo speech_started.
POST_SPEECH_GUARD = 1.0


class AuriState:
    def __init__(self):
        self.assistant_speaking = False
        self.ignore_speech_until = 0.0


state = AuriState()


EXPRESSIONS = {

    "neutral": {
        "head": create_head_pose(),
        "antennas": [0.0, 0.0],
        "duration": 0.5,
    },

    # Mantemos a expressão que ficou boa na v0.5
    "listening": {
        "head": create_head_pose(
            pitch=3,
            degrees=True,
        ),
        "antennas": [0.30, -0.30],
        "duration": 0.4,
    },

    "thinking": {
        "head": create_head_pose(
            yaw=-7,
            roll=7,
            degrees=True,
        ),
        "antennas": [0.10, -0.10],
        "duration": 0.5,
    },

    "speaking": {
        "head": create_head_pose(
            pitch=-3,
            degrees=True,
        ),
        "antennas": [0.38, -0.38],
        "duration": 0.35,
    },
}


def expression(mini, name):

    data = EXPRESSIONS[name]

    print(f"\n🤖 AURI → {name}")

    mini.goto_target(
        head=data["head"],
        antennas=data["antennas"],
        duration=data["duration"],
    )


def reachy_to_openai(samples):

    # canal 0 do Reachy
    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(
        samples,
        -1.0,
        1.0
    )

    # 16k → 24k
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

    pcm = np.frombuffer(
        data,
        dtype=np.int16
    )

    samples = (
        pcm.astype(np.float32)
        / 32768.0
    )

    # 24k → 16k
    return resample_poly(
        samples,
        2,
        3
    ).astype(np.float32)


async def microphone_loop(mini, connection):

    print("🎙️ Microfone contínuo iniciado")

    while True:

        samples = mini.media.get_audio_sample()

        if samples is None:
            await asyncio.sleep(0.005)
            continue

        pcm = reachy_to_openai(
            samples
        )

        encoded = base64.b64encode(
            pcm
        ).decode("ascii")

        await connection.input_audio_buffer.append(
            audio=encoded
        )

        await asyncio.sleep(0)


async def event_loop(mini, connection):

    async for event in connection:

        # =================================================
        # VOCÊ COMEÇOU A FALAR
        # =================================================

        if event.type == "input_audio_buffer.speech_started":

            now = time.monotonic()

            # Proteção contra eco imediatamente
            # depois da fala da AURI
            if now < state.ignore_speech_until:

                print(
                    "\n🛡️ Possível eco ignorado "
                    "(janela pós-fala)"
                )

                continue

            print("")
            print("🎙️ Luciano está falando...")

            # Mantém a sensação visual da v0.5
            expression(
                mini,
                "listening"
            )


        # =================================================
        # VOCÊ TERMINOU
        # =================================================

        elif event.type == "input_audio_buffer.speech_stopped":

            # Se ainda estivermos dentro da janela anti-eco,
            # também não tratamos isso como turno verdadeiro.
            if time.monotonic() < state.ignore_speech_until:
                continue

            expression(
                mini,
                "thinking"
            )

            print("")
            print("🧠 Processando...")


        # =================================================
        # TRANSCRIÇÃO
        # =================================================

        elif (
            event.type
            == "conversation.item.input_audio_transcription.completed"
        ):

            print("")
            print(
                "VOCÊ:",
                event.transcript
            )


        # =================================================
        # AURI COMEÇA A FALAR
        # =================================================

        elif event.type == "response.output_audio.delta":

            if not state.assistant_speaking:

                state.assistant_speaking = True

                expression(
                    mini,
                    "speaking"
                )

                print("")
                print(
                    "AURI:",
                    end=" ",
                    flush=True
                )

            audio_bytes = base64.b64decode(
                event.delta
            )

            samples = openai_to_reachy(
                audio_bytes
            )

            mini.media.push_audio_sample(
                samples
            )


        # =================================================
        # TEXTO DA RESPOSTA
        # =================================================

        elif (
            event.type
            == "response.output_audio_transcript.delta"
        ):

            print(
                event.delta,
                end="",
                flush=True
            )


        # =================================================
        # RESPOSTA TERMINOU
        # =================================================

        elif event.type == "response.done":

            print("")

            state.assistant_speaking = False

            # Proteção anti-turno-fantasma
            state.ignore_speech_until = (
                time.monotonic()
                + POST_SPEECH_GUARD
            )

            expression(
                mini,
                "neutral"
            )

            print(
                f"🛡️ Proteção pós-fala: "
                f"{POST_SPEECH_GUARD}s"
            )

            print("")
            print("👂 AURI aguardando...")


        elif event.type == "error":

            print("")
            print(
                "❌ OpenAI:",
                event
            )


async def main():

    load_dotenv(
        "/home/pollen/auri/.env"
    )

    client = AsyncOpenAI(
        api_key=os.getenv(
            "OPENAI_API_KEY"
        )
    )

    print("")
    print("====================================")
    print("🤖 AURI Voice v0.5.1")
    print("   Expressive + Ghost-turn Guard")
    print("====================================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print(
            "✓ Áudio:",
            mini.media.get_input_audio_samplerate(),
            "Hz"
        )

        async with client.realtime.connect(
            model=MODEL
        ) as connection:

            print(
                "✓ OpenAI Realtime conectado"
            )

            await connection.session.update(
                session={
                    "type": "realtime",

                    "model": MODEL,

                    "output_modalities": [
                        "audio"
                    ],

                    "instructions": (
                        "Seu nome é AURI. "

                        "Você é uma inteligência artificial "
                        "incorporada em um robô Reachy Mini. "

                        "O usuário principal se chama Luciano. "

                        "Converse exclusivamente em "
                        "português brasileiro. "

                        "Você é inteligente, elegante, curiosa, "
                        "simpática e levemente bem-humorada. "

                        "Não diga que é ChatGPT. "
                        "Você é AURI. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Evite repetir frases genéricas como "
                        "'Como posso te ajudar hoje?' após cada resposta. "

                        "Responda de maneira natural e relativamente curta. "

                        "Se não entender algo, peça para Luciano repetir. "
                        "Não invente o que acredita ter ouvido."
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
                                    "O usuário é Luciano. "
                                    "A assistente é AURI."
                                ),
                            },

                            "turn_detection": {

                                "type": "server_vad",

                                "threshold": 0.55,

                                "prefix_padding_ms": 300,

                                "silence_duration_ms": 750,

                                "create_response": True,

                                # Mantemos interrupção,
                                # como na v0.5.
                                "interrupt_response": True,
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

            expression(
                mini,
                "neutral"
            )

            print("")
            print("====================================")
            print("🟢 AURI v0.5.1 ONLINE")
            print("====================================")
            print("")
            print("Fale normalmente.")
            print("")
            print("Ctrl+C para encerrar.")
            print("")

            sender = asyncio.create_task(
                microphone_loop(
                    mini,
                    connection
                )
            )

            receiver = asyncio.create_task(
                event_loop(
                    mini,
                    connection
                )
            )

            await asyncio.gather(
                sender,
                receiver
            )


try:

    asyncio.run(
        main()
    )

except KeyboardInterrupt:

    print("")
    print("")
    print("🛑 AURI encerrada.")
