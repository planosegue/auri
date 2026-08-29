import asyncio
import base64
import io
import os
import time

import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


REALTIME_MODEL = "gpt-realtime-2.1-mini"
VISION_MODEL = "gpt-5.6-luna"

OPENAI_RATE = 24000

POST_SPEECH_GUARD = 0.8


# ============================================================
# ESTADO
# ============================================================

class AuriState:
    def __init__(self):
        self.assistant_speaking = False
        self.vision_processing = False
        self.ignore_speech_until = 0.0

        # Controle da fila de áudio
        self.audio_samples = 0
        self.audio_started_at = None


state = AuriState()


# ============================================================
# EXPRESSÕES
# ============================================================

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

    "curious": {
        "head": create_head_pose(
            yaw=7,
            roll=-8,
            degrees=True,
        ),
        "antennas": [0.35, -0.35],
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


# ============================================================
# ÁUDIO
# ============================================================

def reachy_to_openai(samples):

    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(
        samples,
        -1.0,
        1.0
    )

    # 16 kHz -> 24 kHz
    samples_24k = resample_poly(
        samples,
        3,
        2
    )

    pcm16 = (
        np.clip(
            samples_24k,
            -1.0,
            1.0
        )
        * 32767
    ).astype(np.int16)

    return pcm16.tobytes()


def openai_to_reachy(data):

    pcm = np.frombuffer(
        data,
        dtype=np.int16
    )

    samples_24k = (
        pcm.astype(np.float32)
        / 32768.0
    )

    # 24 kHz -> 16 kHz
    return resample_poly(
        samples_24k,
        2,
        3
    ).astype(np.float32)


# ============================================================
# INTENÇÃO VISUAL
# ============================================================

def is_visual_request(text):

    text = text.lower()

    terms = [
        "o que você está vendo",
        "o que voce esta vendo",
        "o que está vendo",
        "o que esta vendo",
        "você consegue ver",
        "voce consegue ver",
        "olha isso",
        "olhe isso",
        "o que estou segurando",
        "o que eu estou segurando",
        "o que tenho na mão",
        "o que tenho na mao",
        "quantas pessoas",
        "quem está aqui",
        "quem esta aqui",
        "quem está na sua frente",
        "quem esta na sua frente",
        "leia isso",
        "leia o que está",
        "leia o que esta",
        "descreva o ambiente",
        "descreva o que está vendo",
        "descreva o que esta vendo",
        "o que tem aqui",
        "o que tem na sua frente",
    ]

    return any(
        term in text
        for term in terms
    )


# ============================================================
# VISÃO
# ============================================================

async def analyze_vision(
    client,
    mini,
    user_text
):

    state.vision_processing = True

    print("\n👁️ AURI olhando...")

    expression(
        mini,
        "curious"
    )

    await asyncio.sleep(0.6)

    frame = mini.media.get_frame()

    if frame is None:

        state.vision_processing = False

        return (
            "Não consegui enxergar "
            "neste momento."
        )

    print(
        "✓ Frame:",
        frame.shape
    )

    image = Image.fromarray(
        frame
    )

    image.thumbnail(
        (1024, 1024)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=80,
    )

    image_b64 = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    image_url = (
        "data:image/jpeg;base64,"
        + image_b64
    )

    print(
        "🧠 Analisando visão..."
    )

    try:

        response = await client.responses.create(

            model=VISION_MODEL,

            reasoning={
                "effort": "none"
            },

            max_output_tokens=180,

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",

                            "text": (
                                "Você é a percepção visual da AURI, "
                                "uma inteligência incorporada em um "
                                "robô Reachy Mini. "

                                "Responda em português brasileiro. "

                                "Observe apenas aquilo que realmente "
                                "aparece na imagem. "

                                "Não invente nomes, pessoas ou objetos. "

                                "Se não tiver certeza, diga que não "
                                "consegue determinar com segurança. "

                                "Pergunta de Luciano: "
                                + user_text
                            ),
                        },

                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                    ],
                }
            ],
        )

        result = response.output_text.strip()

        print(
            "\n👁️ VISÃO:",
            result
        )

        return result

    except Exception as error:

        print(
            "\n❌ Erro Vision:",
            error
        )

        return (
            "Tive um problema ao analisar "
            "o que estou vendo."
        )

    finally:

        state.vision_processing = False


# ============================================================
# MICROFONE
# ============================================================

async def microphone_loop(
    mini,
    connection
):

    print(
        "🎙️ Microfone contínuo iniciado"
    )

    while True:

        samples = mini.media.get_audio_sample()

        if samples is None:

            await asyncio.sleep(
                0.005
            )

            continue

        # IMPORTANTE:
        # enquanto a AURI fala ou analisa visão,
        # consumimos o áudio local mas NÃO enviamos
        # à OpenAI.

        if (
            state.assistant_speaking
            or state.vision_processing
            or time.monotonic()
                < state.ignore_speech_until
        ):

            await asyncio.sleep(
                0.005
            )

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


# ============================================================
# ESPERA O SPEAKER REALMENTE TERMINAR
# ============================================================

async def wait_for_speaker():

    if state.audio_started_at is None:
        return

    # PCM16 mono a 24 kHz
    total_audio_seconds = (
        state.audio_samples
        / OPENAI_RATE
    )

    elapsed = (
        time.monotonic()
        - state.audio_started_at
    )

    remaining = max(
        0.0,
        total_audio_seconds - elapsed
    )

    # pequena margem
    remaining += 0.20

    if remaining > 0:

        print(
            f"\n🔊 Aguardando speaker: "
            f"{remaining:.2f}s"
        )

        await asyncio.sleep(
            remaining
        )


# ============================================================
# EVENTOS
# ============================================================

async def event_loop(
    mini,
    connection,
    client
):

    async for event in connection:


        # ------------------------------------------------
        # USUÁRIO COMEÇOU A FALAR
        # ------------------------------------------------

        if event.type == \
            "input_audio_buffer.speech_started":

            if (
                state.assistant_speaking
                or state.vision_processing
                or time.monotonic()
                    < state.ignore_speech_until
            ):

                continue

            print(
                "\n🎙️ Luciano está falando..."
            )

            expression(
                mini,
                "listening"
            )


        # ------------------------------------------------
        # USUÁRIO TERMINOU
        # ------------------------------------------------

        elif event.type == \
            "input_audio_buffer.speech_stopped":

            if (
                state.assistant_speaking
                or state.vision_processing
            ):

                continue

            expression(
                mini,
                "thinking"
            )

            print(
                "\n🧠 Processando..."
            )


        # ------------------------------------------------
        # TRANSCRIÇÃO
        # ------------------------------------------------

        elif event.type == \
            "conversation.item.input_audio_transcription.completed":

            if (
                state.assistant_speaking
                or state.vision_processing
            ):

                continue

            user_text = (
                event.transcript
                .strip()
            )

            print(
                "\nVOCÊ:",
                user_text
            )

            if not user_text:
                continue

            # --------------------------------------------
            # VISUAL
            # --------------------------------------------

            if is_visual_request(
                user_text
            ):

                vision_result = await analyze_vision(
                    client,
                    mini,
                    user_text
                )

                expression(
                    mini,
                    "thinking"
                )

                await connection.conversation.item.create(

                    item={

                        "type": "message",

                        "role": "user",

                        "content": [

                            {
                                "type": "input_text",

                                "text": (
                                    "Você acabou de observar o ambiente. "

                                    "A sua percepção visual foi: "
                                    + vision_result +

                                    "\nResponda agora à pergunta "
                                    "anterior de Luciano. "

                                    "Fale como AURI em português "
                                    "brasileiro. "

                                    "Não mencione câmera, API, modelo "
                                    "ou processamento interno."
                                ),
                            }
                        ],
                    }
                )

            await connection.response.create()


        # ------------------------------------------------
        # ÁUDIO DA AURI
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio.delta":

            audio_bytes = base64.b64decode(
                event.delta
            )

            # PCM16 = 2 bytes por amostra
            samples_count = (
                len(audio_bytes)
                // 2
            )

            if not state.assistant_speaking:

                state.assistant_speaking = True

                state.audio_samples = 0

                state.audio_started_at = (
                    time.monotonic()
                )

                expression(
                    mini,
                    "speaking"
                )

                print(
                    "\nAURI:",
                    end=" ",
                    flush=True
                )

            state.audio_samples += (
                samples_count
            )

            samples = openai_to_reachy(
                audio_bytes
            )

            mini.media.push_audio_sample(
                samples
            )


        # ------------------------------------------------
        # TRANSCRIÇÃO DA AURI
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio_transcript.delta":

            print(
                event.delta,
                end="",
                flush=True
            )


        # ------------------------------------------------
        # OPENAI TERMINOU DE GERAR
        # ------------------------------------------------

        elif event.type == \
            "response.done":

            print("")

            # MUITO IMPORTANTE:
            # response.done NÃO significa que o
            # speaker terminou.

            await wait_for_speaker()

            state.assistant_speaking = False

            state.audio_samples = 0
            state.audio_started_at = None

            state.ignore_speech_until = (
                time.monotonic()
                + POST_SPEECH_GUARD
            )

            expression(
                mini,
                "neutral"
            )

            print(
                f"🛡️ Anti-eco: "
                f"{POST_SPEECH_GUARD}s"
            )

            print(
                "\n👂 AURI aguardando..."
            )


        elif event.type == "error":

            print(
                "\n❌ OpenAI:",
                event
            )


# ============================================================
# MAIN
# ============================================================

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
    print(
        "===================================="
    )
    print(
        "🤖 AURI v0.5.2.1"
    )
    print(
        "   Voice + Vision Stable"
    )
    print(
        "===================================="
    )
    print("")

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print(
            "✓ Reachy conectado"
        )

        mini.media.start_recording()
        mini.media.start_playing()

        print(
            "✓ Áudio:",
            mini.media.get_input_audio_samplerate(),
            "Hz"
        )

        frame = mini.media.get_frame()

        if frame is not None:

            print(
                "✓ Visão:",
                frame.shape
            )

        async with client.realtime.connect(
            model=REALTIME_MODEL
        ) as connection:

            print(
                "✓ OpenAI Realtime conectado"
            )

            await connection.session.update(

                session={

                    "type": "realtime",

                    "model": REALTIME_MODEL,

                    "output_modalities": [
                        "audio"
                    ],

                    "instructions": (
                        "Seu nome é AURI. "

                        "Você é uma inteligência artificial "
                        "incorporada fisicamente em um Reachy Mini. "

                        "Luciano é seu usuário principal. "

                        "Fale exclusivamente em português brasileiro. "

                        "Seja inteligente, elegante, curiosa, "
                        "simpática e levemente bem-humorada. "

                        "Não diga que é ChatGPT. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Não repita constantemente frases como "
                        "'Como posso te ajudar hoje?'. "

                        "Respostas de conversa devem ser "
                        "relativamente curtas. "

                        "Quando receber uma percepção visual, "
                        "trate-a como algo que você própria "
                        "acabou de observar. "

                        "Nunca invente o que não ouviu ou viu."
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
                                    "Usuário: Luciano. "
                                    "Assistente: AURI."
                                ),
                            },

                            "turn_detection": {

                                "type": "server_vad",

                                "threshold": 0.55,

                                "prefix_padding_ms": 300,

                                "silence_duration_ms": 750,

                                "create_response": False,

                                # Por enquanto priorizamos
                                # estabilidade.
                                "interrupt_response": False,
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
            print(
                "===================================="
            )
            print(
                "🟢 AURI v0.5.2.1 ONLINE"
            )
            print(
                "===================================="
            )
            print("")
            print(
                "Voz + visão estável."
            )
            print("")
            print(
                "Espere AURI terminar de falar "
                "antes de fazer nova pergunta."
            )
            print("")
            print(
                "Teste:"
            )
            print("")
            print(
                "  Auri, o que você está vendo?"
            )
            print("")
            print(
                "  Auri, o que eu estou segurando?"
            )
            print("")
            print(
                "Ctrl+C para encerrar."
            )
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
                    connection,
                    client
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
    print(
        "🛑 AURI encerrada."
    )
