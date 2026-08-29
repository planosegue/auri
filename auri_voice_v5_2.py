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


# ============================================================
# AURI v0.5.2
# Voice + Vision
# ============================================================

REALTIME_MODEL = "gpt-realtime-2.1-mini"
VISION_MODEL = "gpt-5.6-luna"

OPENAI_RATE = 24000

POST_SPEECH_GUARD = 1.0


# ============================================================
# ESTADO
# ============================================================

class AuriState:

    def __init__(self):

        self.assistant_speaking = False

        self.ignore_speech_until = 0.0

        self.waiting_for_response = False


state = AuriState()


# ============================================================
# EXPRESSÕES
# ============================================================

EXPRESSIONS = {

    "neutral": {

        "head": create_head_pose(),

        "antennas": [
            0.0,
            0.0
        ],

        "duration": 0.5,
    },


    "listening": {

        "head": create_head_pose(
            pitch=3,
            degrees=True
        ),

        "antennas": [
            0.30,
            -0.30
        ],

        "duration": 0.4,
    },


    "thinking": {

        "head": create_head_pose(
            yaw=-7,
            roll=7,
            degrees=True
        ),

        "antennas": [
            0.10,
            -0.10
        ],

        "duration": 0.5,
    },


    "curious": {

        "head": create_head_pose(
            yaw=7,
            roll=-8,
            degrees=True
        ),

        "antennas": [
            0.35,
            -0.35
        ],

        "duration": 0.5,
    },


    "speaking": {

        "head": create_head_pose(
            pitch=-3,
            degrees=True
        ),

        "antennas": [
            0.38,
            -0.38
        ],

        "duration": 0.35,
    },
}


def expression(mini, name):

    data = EXPRESSIONS[name]

    print(
        f"\n🤖 AURI → {name}"
    )

    mini.goto_target(

        head=data["head"],

        antennas=data["antennas"],

        duration=data["duration"],
    )


# ============================================================
# ÁUDIO
# ============================================================

def reachy_to_openai(samples):

    # Reachy:
    # float32
    # stereo
    # 16 kHz

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

    ).astype(
        np.int16
    )


    return pcm16.tobytes()


def openai_to_reachy(data):

    pcm = np.frombuffer(

        data,

        dtype=np.int16
    )


    samples_24k = (

        pcm.astype(
            np.float32
        )

        / 32768.0
    )


    # 24 kHz -> 16 kHz

    samples_16k = resample_poly(

        samples_24k,

        2,

        3
    )


    return samples_16k.astype(
        np.float32
    )


# ============================================================
# DETECÇÃO DE INTENÇÃO VISUAL
# ============================================================

def is_visual_request(text):

    text = text.lower()


    visual_terms = [

        "o que você está vendo",

        "o que voce esta vendo",

        "o que está vendo",

        "o que esta vendo",

        "você consegue ver",

        "voce consegue ver",

        "olha isso",

        "olhe isso",

        "olha para isso",

        "o que eu estou segurando",

        "o que estou segurando",

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

        "o que tem na sua frente",

        "o que tem aqui",
    ]


    return any(

        term in text

        for term in visual_terms
    )


# ============================================================
# VISÃO
# ============================================================

async def analyze_vision(
    client,
    mini,
    user_text
):

    print("")
    print("👁️ AURI ativando visão...")


    expression(
        mini,
        "curious"
    )


    # Pequeno tempo para o movimento terminar
    # antes de capturar a imagem.

    await asyncio.sleep(
        0.6
    )


    frame = mini.media.get_frame()


    if frame is None:

        print(
            "❌ Nenhum frame recebido"
        )

        return (
            "Não consegui enxergar "
            "neste momento."
        )


    print(
        "✓ Frame:",
        frame.shape
    )


    # Reachy entrega numpy uint8.

    image = Image.fromarray(
        frame
    )


    # Reduz um pouco a imagem.
    # Economiza banda/tokens sem perder
    # muita informação visual.

    image.thumbnail(
        (1024, 1024)
    )


    buffer = io.BytesIO()


    image.save(

        buffer,

        format="JPEG",

        quality=80
    )


    image_b64 = base64.b64encode(

        buffer.getvalue()

    ).decode(
        "ascii"
    )


    image_url = (

        "data:image/jpeg;base64,"

        + image_b64
    )


    print(
        "🧠 Analisando imagem..."
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
                                "Você é o sistema visual da AURI, "
                                "uma inteligência artificial incorporada "
                                "em um robô Reachy Mini. "

                                "Analise a imagem atual capturada pelos "
                                "olhos do robô. "

                                "Responda em português brasileiro. "

                                "Descreva apenas aquilo que realmente "
                                "pode ser observado. "

                                "Não invente nomes de pessoas, textos, "
                                "objetos ou detalhes que não estejam "
                                "claros. "

                                "Se não tiver certeza, diga que não "
                                "consegue determinar com segurança. "

                                "Seja objetiva. "

                                "A pergunta do usuário foi: "
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


        print("")
        print(
            "👁️ VISÃO:",
            result
        )


        return result


    except Exception as error:

        print("")
        print(
            "❌ Erro Vision:",
            error
        )


        return (
            "Tive um problema ao analisar "
            "o que estou vendo."
        )


# ============================================================
# MICROFONE CONTÍNUO
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


        pcm = reachy_to_openai(
            samples
        )


        encoded = base64.b64encode(

            pcm

        ).decode(
            "ascii"
        )


        await connection.input_audio_buffer.append(

            audio=encoded
        )


        await asyncio.sleep(
            0
        )


# ============================================================
# EVENT LOOP
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


            now = time.monotonic()


            if now < state.ignore_speech_until:

                print("")
                print(
                    "🛡️ Possível eco ignorado"
                )

                continue


            print("")
            print(
                "🎙️ Luciano está falando..."
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
                time.monotonic()
                < state.ignore_speech_until
            ):

                continue


            expression(
                mini,
                "thinking"
            )


            print("")
            print(
                "🧠 Processando..."
            )


        # ------------------------------------------------
        # TRANSCRIÇÃO DO USUÁRIO
        # ------------------------------------------------

        elif event.type == \
            "conversation.item.input_audio_transcription.completed":


            user_text = event.transcript.strip()


            print("")
            print(
                "VOCÊ:",
                user_text
            )


            if not user_text:

                continue


            # ============================================
            # PERGUNTA VISUAL
            # ============================================

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
                                    "Você acabou de olhar para o ambiente. "

                                    "Esta foi a informação obtida pela "
                                    "sua visão: "

                                    + vision_result +

                                    "\n\nResponda agora à pergunta "
                                    "anterior do Luciano. "

                                    "Fale naturalmente em português "
                                    "brasileiro, como AURI. "

                                    "Não mencione APIs, modelos, câmera, "
                                    "processamento de imagem ou sistemas "
                                    "internos. "

                                    "Fale como se você própria tivesse "
                                    "acabado de observar isso."
                                ),
                            }
                        ],
                    }
                )


                state.waiting_for_response = True


                await connection.response.create()


            # ============================================
            # PERGUNTA NORMAL
            # ============================================

            else:

                state.waiting_for_response = True


                await connection.response.create()


        # ------------------------------------------------
        # ÁUDIO DA AURI
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio.delta":


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


        # ------------------------------------------------
        # TEXTO DA RESPOSTA
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio_transcript.delta":


            print(

                event.delta,

                end="",

                flush=True
            )


        # ------------------------------------------------
        # RESPOSTA TERMINOU
        # ------------------------------------------------

        elif event.type == \
            "response.done":


            print("")


            state.assistant_speaking = False

            state.waiting_for_response = False


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
            print(
                "👂 AURI aguardando..."
            )


        # ------------------------------------------------
        # ERROS
        # ------------------------------------------------

        elif event.type == "error":


            print("")
            print(
                "❌ OpenAI:",
                event
            )


# ============================================================
# MAIN
# ============================================================

async def main():


    load_dotenv(
        "/home/pollen/auri/.env"
    )


    api_key = os.getenv(
        "OPENAI_API_KEY"
    )


    client = AsyncOpenAI(
        api_key=api_key
    )


    print("")
    print(
        "===================================="
    )
    print(
        "🤖 AURI Voice v0.5.2"
    )
    print(
        "   Realtime + Vision"
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

        else:

            print(
                "⚠️ Câmera sem frame inicial"
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
                        "incorporada fisicamente em um "
                        "robô Reachy Mini. "

                        "O usuário principal se chama Luciano. "

                        "Converse exclusivamente em "
                        "português brasileiro. "

                        "Você é inteligente, elegante, curiosa, "
                        "simpática e levemente bem-humorada. "

                        "Não diga que é ChatGPT. "
                        "Você é AURI. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Evite frases repetitivas como "
                        "'Como posso te ajudar hoje?'. "

                        "Responda de maneira natural e "
                        "relativamente curta. "

                        "Você possui visão através dos olhos "
                        "do seu corpo robótico. "

                        "Quando receber informação do seu "
                        "sistema visual, trate essa informação "
                        "como aquilo que você acabou de observar. "

                        "Se não entender algo, peça para "
                        "Luciano repetir. "

                        "Não invente informações."
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

                                # IMPORTANTE:
                                # agora nós decidimos quando
                                # gerar a resposta.

                                "create_response": False,

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
            print(
                "===================================="
            )
            print(
                "🟢 AURI v0.5.2 ONLINE"
            )
            print(
                "===================================="
            )
            print("")
            print(
                "Voz + visão disponíveis."
            )
            print("")
            print(
                "Experimente:"
            )
            print("")
            print(
                "  Auri, tudo bem?"
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
                "  Auri, quantas pessoas estão aqui?"
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
