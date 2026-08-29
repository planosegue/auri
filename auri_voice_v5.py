import asyncio
import base64
import os

import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import AsyncOpenAI

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


MODEL = "gpt-realtime-2.1-mini"

REACHY_RATE = 16000
OPENAI_RATE = 24000


# =========================================================
# EXPRESSÕES
# =========================================================

EXPRESSIONS = {
    "neutral": {
        "head": create_head_pose(),
        "antennas": [0.0, 0.0],
        "duration": 0.5,
    },

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

    state = EXPRESSIONS[name]

    print(f"\n🤖 AURI → {name}")

    mini.goto_target(
        head=state["head"],
        antennas=state["antennas"],
        duration=state["duration"],
    )


# =========================================================
# ÁUDIO
# =========================================================

def reachy_to_openai(samples):

    # Reachy: float32 estéreo 16 kHz
    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(samples, -1.0, 1.0)

    # 16 kHz → 24 kHz
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

    # 24 kHz → 16 kHz
    samples_16k = resample_poly(
        samples_24k,
        2,
        3
    ).astype(np.float32)

    return samples_16k


# =========================================================
# ENVIO CONTÍNUO DO MICROFONE
# =========================================================

async def microphone_loop(mini, connection):

    print("🎙️ Microfone contínuo iniciado")

    while True:

        samples = mini.media.get_audio_sample()

        if samples is None:
            await asyncio.sleep(0.005)
            continue

        pcm = reachy_to_openai(samples)

        encoded = base64.b64encode(
            pcm
        ).decode("ascii")

        await connection.input_audio_buffer.append(
            audio=encoded
        )

        await asyncio.sleep(0)


# =========================================================
# EVENTOS REALTIME
# =========================================================

async def event_loop(mini, connection):

    speaking = False

    async for event in connection:

        # ------------------------------
        # Usuário começou a falar
        # ------------------------------

        if event.type == "input_audio_buffer.speech_started":

            speaking = False

            expression(
                mini,
                "listening"
            )

            print("\n🎙️ Luciano está falando...")


        # ------------------------------
        # Usuário terminou
        # ------------------------------

        elif event.type == "input_audio_buffer.speech_stopped":

            expression(
                mini,
                "thinking"
            )

            print("\n🧠 Processando...")


        # ------------------------------
        # Transcrição do usuário
        # ------------------------------

        elif (
            event.type
            == "conversation.item.input_audio_transcription.completed"
        ):

            print(
                "\nVOCÊ:",
                event.transcript
            )


        # ------------------------------
        # Áudio da resposta
        # ------------------------------

        elif event.type == "response.output_audio.delta":

            if not speaking:

                speaking = True

                expression(
                    mini,
                    "speaking"
                )

                print("\nAURI:", end=" ", flush=True)

            audio_bytes = base64.b64decode(
                event.delta
            )

            samples = openai_to_reachy(
                audio_bytes
            )

            mini.media.push_audio_sample(
                samples
            )


        # ------------------------------
        # Texto da fala da AURI
        # ------------------------------

        elif (
            event.type
            == "response.output_audio_transcript.delta"
        ):

            print(
                event.delta,
                end="",
                flush=True
            )


        # ------------------------------
        # Resposta terminou
        # ------------------------------

        elif event.type == "response.done":

            print("")

            speaking = False

            expression(
                mini,
                "neutral"
            )

            print("\n👂 AURI aguardando...")


        # ------------------------------
        # Erros
        # ------------------------------

        elif event.type == "error":

            print(
                "\n❌ OpenAI:",
                event
            )


# =========================================================
# MAIN
# =========================================================

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
    print("🤖 AURI Voice v0.5")
    print("   Conversa contínua + VAD")
    print("====================================")
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✓ Reachy conectado")

        mini.media.start_recording()
        mini.media.start_playing()

        print(
            "✓ Áudio Reachy:",
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
                        "Seu nome é AURI. "
                        "Você é uma inteligência artificial "
                        "incorporada fisicamente em um robô Reachy Mini. "

                        "O usuário principal se chama Luciano. "

                        "Converse sempre em português brasileiro. "
                        "Nunca responda em outro idioma, exceto se Luciano "
                        "pedir explicitamente. "

                        "Você é inteligente, elegante, curiosa, "
                        "simpática e levemente bem-humorada. "

                        "Sua conversa deve parecer natural e humana. "

                        "Não diga que é ChatGPT. "
                        "Você é AURI. "

                        "Prefira respostas curtas durante conversas comuns. "

                        "Se não entender claramente alguma coisa, "
                        "pergunte se Luciano pode repetir. "

                        "Nunca invente algo que não conseguiu ouvir."
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
                                    "O usuário se chama Luciano. "
                                    "A assistente se chama AURI."
                                ),
                            },

                            "turn_detection": {

                                "type": "server_vad",

                                # Detecta início de voz
                                "threshold": 0.5,

                                # Pequeno áudio anterior ao início
                                "prefix_padding_ms": 300,

                                # Considera que terminou depois
                                # deste período de silêncio
                                "silence_duration_ms": 700,

                                # Gera resposta automaticamente
                                "create_response": True,

                                # Permite interromper AURI
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
            print("🟢 AURI ONLINE")
            print("====================================")
            print("")
            print("Fale naturalmente.")
            print("Exemplo:")
            print("")
            print("   Auri, tudo bem?")
            print("")
            print("Pressione Ctrl+C para encerrar.")
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
