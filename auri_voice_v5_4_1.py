import asyncio
import base64
import io
import json
import os
import re
import subprocess
import time

import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


# ============================================================
# AURI v0.5.4.1
# Real Tools + Stable Async + Automatic Barge-in
# ============================================================

REALTIME_MODEL = "gpt-realtime-2.1-mini"
VISION_MODEL = "gpt-5.6-luna"
WEB_MODEL = "gpt-5.6-luna"

OPENAI_RATE = 24000

MIN_VOLUME = 10
MAX_VOLUME = 100
VOLUME_STEP = 15

# Evita repouso exatamente em 0°, que provocava tremor.
ANTENNA_REST_OFFSET = 0.17

# Proteção curta depois que o speaker termina.
POST_SPEECH_GUARD = 0.45

# Ignora apenas a limpeza física nos primeiros instantes da fala,
# evitando reagir a pequenos estalos do próprio início do playback.
BARGE_IN_ARM_DELAY = 0.50


# ============================================================
# ESTADO
# ============================================================

class AuriState:

    def __init__(self):

        self.assistant_speaking = False
        self.assistant_started_at = None

        self.user_speaking = False

        self.vision_processing = False
        self.web_processing = False

        self.ignore_speech_until = 0.0

        self.audio_samples = 0
        self.audio_started_at = None

        self.current_volume = 80
        self.current_expression = None

        # Identifica gerações diferentes de playback.
        self.playback_generation = 0
        self.finalize_task = None

        # Function calling
        self.tool_call_active = False
        self.tool_result_ready = False
        self.tool_origin_done = False
        self.tool_followup_started = False
        self.tool_task = None


state = AuriState()


# ============================================================
# EXPRESSÕES
# ============================================================

EXPRESSIONS = {

    "neutral": {
        "head": create_head_pose(),
        "antennas": [
            ANTENNA_REST_OFFSET,
            -ANTENNA_REST_OFFSET,
        ],
        "duration": 0.5,
    },

    "listening": {
        "head": create_head_pose(
            pitch=3,
            degrees=True,
        ),
        "antennas": [
            0.30,
            -0.30,
        ],
        "duration": 0.35,
    },

    "thinking": {
        "head": create_head_pose(
            yaw=-7,
            roll=7,
            degrees=True,
        ),
        "antennas": [
            0.12,
            -0.12,
        ],
        "duration": 0.45,
    },

    "curious": {
        "head": create_head_pose(
            yaw=7,
            roll=-8,
            degrees=True,
        ),
        "antennas": [
            0.35,
            -0.35,
        ],
        "duration": 0.45,
    },

    "speaking": {
        "head": create_head_pose(
            pitch=-3,
            degrees=True,
        ),
        "antennas": [
            0.38,
            -0.38,
        ],
        "duration": 0.30,
    },
}


def expression(mini, name):

    if state.current_expression == name:
        return

    state.current_expression = name

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

    # Reachy = stereo float32 16 kHz.
    # Canal 0 foi o pipeline validado.
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
    samples_16k = resample_poly(
        samples_24k,
        2,
        3
    )

    return samples_16k.astype(
        np.float32
    )


# ============================================================
# PLAYBACK
# ============================================================

def cancel_finalize_task():

    task = state.finalize_task

    if (
        task is not None
        and not task.done()
    ):

        task.cancel()

    state.finalize_task = None


def reset_playback_state():

    state.assistant_speaking = False
    state.assistant_started_at = None

    state.audio_samples = 0
    state.audio_started_at = None


def clear_reachy_player(mini):

    try:

        # MediaManager guarda o backend de áudio aqui.
        audio = getattr(
            mini.media,
            "audio",
            None
        )

        if (
            audio is not None
            and hasattr(
                audio,
                "clear_player"
            )
        ):

            audio.clear_player()

            print(
                "🧹 Speaker buffer limpo"
            )

            return True


        print(
            "⚠️ clear_player não disponível"
        )

        return False


    except Exception as error:

        print(
            "⚠️ Erro clear_player:",
            repr(error)
        )

        return False


async def finalize_playback(
    mini,
    generation
):

    """
    Espera o speaker terminar SEM bloquear
    o event_loop do WebSocket.
    """

    try:

        if state.audio_started_at is None:
            return


        total_seconds = (
            state.audio_samples
            / OPENAI_RATE
        )


        elapsed = (
            time.monotonic()
            - state.audio_started_at
        )


        remaining = max(
            0.0,
            total_seconds - elapsed
        )

        remaining += 0.20


        if remaining > 0:

            print(
                f"\n🔊 Speaker restante: "
                f"{remaining:.2f}s"
            )

            await asyncio.sleep(
                remaining
            )


        # Se houve barge-in ou começou outra geração,
        # esta tarefa ficou obsoleta.
        if (
            generation
            != state.playback_generation
        ):

            return


        if state.user_speaking:
            return


        reset_playback_state()


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


    except asyncio.CancelledError:

        return


# ============================================================
# VOLUME
# ============================================================

def get_volume():

    try:

        result = subprocess.run(
            [
                "amixer",
                "-c",
                "0",
                "get",
                "PCM,0",
            ],
            capture_output=True,
            text=True,
            check=True,
        )


        values = re.findall(
            r"\[(\d+)%\]",
            result.stdout
        )


        if values:

            volume = int(
                values[0]
            )

            state.current_volume = volume

            return volume


    except Exception as error:

        print(
            "\n⚠️ Leitura volume:",
            repr(error)
        )


    return state.current_volume


def set_volume_percent(percent):

    percent = int(
        max(
            MIN_VOLUME,
            min(
                MAX_VOLUME,
                percent
            )
        )
    )


    try:

        subprocess.run(
            [
                "amixer",
                "-c",
                "0",
                "sset",
                "PCM,0",
                f"{percent}%",
                "unmute",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )


        state.current_volume = percent


        print(
            f"\n🔊 Volume físico: {percent}%"
        )


        return {
            "ok": True,
            "volume": percent,
        }


    except Exception as error:

        return {
            "ok": False,
            "error": str(error),
        }


def execute_volume_tool(args):

    action = args.get(
        "action",
        "set"
    )

    current = get_volume()


    if action == "up":

        target = min(
            MAX_VOLUME,
            current + VOLUME_STEP
        )


    elif action == "down":

        target = max(
            MIN_VOLUME,
            current - VOLUME_STEP
        )


    else:

        target = args.get(
            "percent",
            current
        )


    return set_volume_percent(
        target
    )


# ============================================================
# WEB SEARCH
# ============================================================

async def search_web(
    client,
    query
):

    state.web_processing = True

    print("")
    print(
        f"🌐 Pesquisando: {query}"
    )


    try:

        response = await client.responses.create(

            model=WEB_MODEL,

            tools=[
                {
                    "type": "web_search"
                }
            ],

            tool_choice="required",

            reasoning={
                "effort": "low"
            },

            max_output_tokens=350,

            input=(
                "Pesquise na internet a solicitação abaixo. "
                "Priorize fabricantes, documentação oficial, "
                "fontes técnicas e informações recentes. "
                "Responda em português brasileiro. "
                "Seja factual e conciso. "
                "Não invente dados.\n\n"
                f"Consulta: {query}"
            ),
        )


        result = (
            response.output_text
            .strip()
        )


        print("")
        print(
            "🌐 RESULTADO:"
        )

        print(
            result
        )


        return {
            "ok": True,
            "result": result,
        }


    except Exception as error:

        print(
            "\n❌ Web:",
            repr(error)
        )


        return {
            "ok": False,
            "error": str(error),
        }


    finally:

        state.web_processing = False


# ============================================================
# VISÃO
# ============================================================

async def look(
    client,
    mini,
    question
):

    state.vision_processing = True


    print("")
    print(
        f"👁️ AURI olhando: {question}"
    )


    expression(
        mini,
        "curious"
    )


    await asyncio.sleep(
        0.50
    )


    frame = mini.media.get_frame()


    if frame is None:

        state.vision_processing = False

        return {
            "ok": False,
            "error": "Nenhum frame recebido.",
        }


    print(
        "✓ Frame:",
        frame.shape
    )


    # Reachy fornece BGR.
    frame_rgb = frame[:, :, ::-1]


    image = Image.fromarray(
        frame_rgb
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
    ).decode(
        "ascii"
    )


    image_url = (
        "data:image/jpeg;base64,"
        + image_b64
    )


    try:

        response = await client.responses.create(

            model=VISION_MODEL,

            reasoning={
                "effort": "none"
            },

            max_output_tokens=130,

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",

                            "text": (
                                "Você é a percepção visual da AURI. "
                                "Analise apenas o que realmente está "
                                "visível. "
                                "Não invente detalhes. "
                                "Responda em português brasileiro. "
                                "Produza uma percepção CURTA e objetiva, "
                                "normalmente em duas ou três frases, "
                                "para uso em uma conversa falada.\n\n"
                                f"Pergunta: {question}"
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


        result = (
            response.output_text
            .strip()
        )


        print("")
        print(
            "👁️ PERCEPÇÃO:"
        )

        print(
            result
        )


        return {
            "ok": True,
            "result": result,
        }


    except Exception as error:

        print(
            "\n❌ Vision:",
            repr(error)
        )


        return {
            "ok": False,
            "error": str(error),
        }


    finally:

        state.vision_processing = False


# ============================================================
# EXECUTOR DE TOOLS
# ============================================================

async def execute_tool(
    client,
    mini,
    name,
    arguments
):

    print("")
    print(
        f"🛠️ TOOL → {name}"
    )


    try:

        args = json.loads(
            arguments
        )

    except Exception:

        args = {}


    print(
        "   Args:",
        args
    )


    if name == "search_web":

        return await search_web(
            client,
            args.get(
                "query",
                ""
            )
        )


    if name == "look":

        return await look(
            client,
            mini,
            args.get(
                "question",
                "O que está na minha frente?"
            )
        )


    if name == "set_volume":

        return execute_volume_tool(
            args
        )


    return {
        "ok": False,
        "error": (
            f"Ferramenta desconhecida: {name}"
        ),
    }


# ============================================================
# TOOL FOLLOW-UP
# ============================================================

async def maybe_start_tool_followup(
    connection
):

    if not state.tool_call_active:
        return


    if state.tool_followup_started:
        return


    if not state.tool_result_ready:
        return


    if not state.tool_origin_done:
        return


    state.tool_followup_started = True


    print("")
    print(
        "🧠 Gerando resposta com resultado da tool..."
    )


    await connection.response.create()


async def process_tool_call(
    client,
    mini,
    connection,
    name,
    arguments,
    call_id
):

    """
    Executa web/visão fora do event_loop.
    O socket continua sendo consumido normalmente.
    """

    try:

        result = await execute_tool(
            client,
            mini,
            name,
            arguments
        )


        await connection.conversation.item.create(

            item={
                "type": "function_call_output",

                "call_id": call_id,

                "output": json.dumps(
                    result,
                    ensure_ascii=False
                ),
            }
        )


        state.tool_result_ready = True


        print(
            "✓ Resultado da tool enviado"
        )


        await maybe_start_tool_followup(
            connection
        )


    except Exception as error:

        print(
            "\n❌ Tool task:",
            repr(error)
        )


        state.tool_result_ready = True


        try:

            await connection.conversation.item.create(

                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {
                            "ok": False,
                            "error": str(error),
                        },
                        ensure_ascii=False
                    ),
                }
            )


            await maybe_start_tool_followup(
                connection
            )


        except Exception as nested_error:

            print(
                "❌ Falha ao enviar erro da tool:",
                repr(nested_error)
            )


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


        # Enquanto web/vision processam, descartamos
        # áudio para não criar turnos acidentais.
        #
        # Enquanto AURI FALA, o microfone continua ativo:
        # isso é necessário para barge-in.
        if (
            state.vision_processing
            or state.web_processing
            or (
                not state.assistant_speaking
                and time.monotonic()
                < state.ignore_speech_until
            )
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
# REALTIME EVENT LOOP
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


            state.user_speaking = True


            if state.assistant_speaking:

                elapsed = 999.0

                if state.assistant_started_at is not None:

                    elapsed = (
                        time.monotonic()
                        - state.assistant_started_at
                    )


                print("")
                print(
                    "=============================="
                )
                print(
                    "✋ BARGE-IN DETECTADO"
                )
                print(
                    "=============================="
                )


                # O servidor já cancela a resposta porque
                # interrupt_response=True.
                #
                # Nosso trabalho é parar o hardware físico.
                if elapsed >= BARGE_IN_ARM_DELAY:

                    cancel_finalize_task()

                    state.playback_generation += 1

                    clear_reachy_player(
                        mini
                    )

                    reset_playback_state()


                expression(
                    mini,
                    "listening"
                )


                print(
                    "🎙️ Ouvindo Luciano..."
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
        # USUÁRIO PAROU
        # ------------------------------------------------

        elif event.type == \
            "input_audio_buffer.speech_stopped":


            state.user_speaking = False


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


            transcript = (
                event.transcript
                .strip()
            )


            if transcript:

                print(
                    "\nVOCÊ:",
                    transcript
                )


        # ------------------------------------------------
        # FUNCTION CALL
        # ------------------------------------------------

        elif event.type == \
            "response.function_call_arguments.done":


            print("")
            print(
                f"🎯 TOOL escolhida: "
                f"{event.name}"
            )


            # Caso o modelo tenha começado a falar alguma
            # frase provisória antes da tool, descartamos.
            if state.assistant_speaking:

                cancel_finalize_task()

                state.playback_generation += 1

                clear_reachy_player(
                    mini
                )

                reset_playback_state()


            state.tool_call_active = True
            state.tool_result_ready = False
            state.tool_origin_done = False
            state.tool_followup_started = False


            expression(
                mini,
                "thinking"
            )


            # MUITO IMPORTANTE:
            # não usamos "await execute_tool" aqui.
            #
            # A pesquisa/visão roda numa task paralela,
            # mantendo o WebSocket sendo lido.
            state.tool_task = asyncio.create_task(

                process_tool_call(
                    client,
                    mini,
                    connection,
                    event.name,
                    event.arguments,
                    event.call_id
                )
            )


        # ------------------------------------------------
        # ÁUDIO DA AURI
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio.delta":


            audio_bytes = base64.b64decode(
                event.delta
            )


            samples_count = (
                len(audio_bytes)
                // 2
            )


            if not state.assistant_speaking:

                state.assistant_speaking = True

                state.assistant_started_at = (
                    time.monotonic()
                )

                state.audio_samples = 0

                state.audio_started_at = (
                    time.monotonic()
                )


                state.playback_generation += 1


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
        # TRANSCRIÇÃO DA VOZ
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio_transcript.delta":


            print(
                event.delta,
                end="",
                flush=True
            )


        # ------------------------------------------------
        # RESPONSE DONE
        # ------------------------------------------------

        elif event.type == \
            "response.done":


            status = getattr(
                event.response,
                "status",
                None
            )


            # --------------------------------------------
            # BARGE-IN / CANCELAMENTO AUTOMÁTICO
            # --------------------------------------------

            if status == "cancelled":

                print("")
                print(
                    "✋ Resposta anterior cancelada"
                )


                cancel_finalize_task()

                state.playback_generation += 1

                reset_playback_state()


                if state.user_speaking:

                    expression(
                        mini,
                        "listening"
                    )


                continue


            # --------------------------------------------
            # RESPOSTA QUE GEROU TOOL CALL
            # --------------------------------------------

            if (
                state.tool_call_active
                and not state.tool_followup_started
            ):

                state.tool_origin_done = True


                await maybe_start_tool_followup(
                    connection
                )


                continue


            # --------------------------------------------
            # RESPOSTA FINAL APÓS TOOL
            # --------------------------------------------

            if (
                state.tool_call_active
                and state.tool_followup_started
            ):

                state.tool_call_active = False
                state.tool_result_ready = False
                state.tool_origin_done = False
                state.tool_followup_started = False
                state.tool_task = None


            # --------------------------------------------
            # FINALIZAÇÃO FÍSICA NÃO-BLOQUEANTE
            # --------------------------------------------

            if state.assistant_speaking:

                generation = (
                    state.playback_generation
                )


                cancel_finalize_task()


                state.finalize_task = asyncio.create_task(

                    finalize_playback(
                        mini,
                        generation
                    )
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
# TOOLS REALTIME
# ============================================================

TOOLS = [

    {
        "type": "function",

        "name": "search_web",

        "description": (
            "Pesquisa a internet quando a pergunta depende "
            "de informação atual ou recente, como notícias, "
            "preços, lançamentos, firmware, documentação online "
            "ou quando Luciano pedir explicitamente para pesquisar. "
            "Não use para conhecimento histórico ou conceitual "
            "estável que você já sabe responder."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "query": {
                    "type": "string",
                    "description": (
                        "Consulta clara e completa para a web."
                    ),
                }
            },

            "required": [
                "query"
            ],

            "additionalProperties": False,
        },
    },


    {
        "type": "function",

        "name": "look",

        "description": (
            "Usa a câmera física da AURI para enxergar "
            "o ambiente atual. Use quando a resposta depende "
            "de ver pessoas, objetos, cores, textos ou a cena."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "question": {
                    "type": "string",
                    "description": (
                        "Pergunta visual a ser respondida."
                    ),
                }
            },

            "required": [
                "question"
            ],

            "additionalProperties": False,
        },
    },


    {
        "type": "function",

        "name": "set_volume",

        "description": (
            "Controla o volume físico do speaker. "
            "Use action='up' para aumentar, "
            "action='down' para diminuir, "
            "e action='set' com percent para nível específico."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "action": {
                    "type": "string",
                    "enum": [
                        "set",
                        "up",
                        "down"
                    ],
                },

                "percent": {
                    "type": "integer",
                    "minimum": MIN_VOLUME,
                    "maximum": MAX_VOLUME,
                },
            },

            "required": [
                "action"
            ],

            "additionalProperties": False,
        },
    },
]


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


    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY não encontrada."
        )


    client = AsyncOpenAI(
        api_key=api_key
    )


    current_volume = get_volume()


    print("")
    print(
        "===================================="
    )
    print(
        "🤖 AURI v0.5.4.1"
    )
    print(
        " Real Tools + Stable Barge-in"
    )
    print(
        "===================================="
    )
    print("")
    print(
        f"🔊 Volume atual: {current_volume}%"
    )
    print(
        f"📡 Antenna rest: "
        f"{ANTENNA_REST_OFFSET} rad"
    )


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

                        "Seja inteligente, elegante, curiosa, simpática "
                        "e levemente bem-humorada. "

                        "Não diga que é ChatGPT. Você é AURI. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Não repita constantemente frases como "
                        "'como posso te ajudar hoje?'. "

                        "Prefira respostas naturais e concisas, "
                        "normalmente de uma a três frases. "

                        "Se Luciano pedir detalhes, então pode aprofundar. "

                        "Você possui ferramentas reais. "

                        "Use search_web somente quando a resposta depender "
                        "de informação atual, recente, online, ou quando "
                        "Luciano pedir explicitamente uma pesquisa. "

                        "Não use internet desnecessariamente para fatos "
                        "históricos ou conhecimento estável. "

                        "Use look quando precisar enxergar algo no ambiente. "

                        "Use set_volume quando Luciano pedir alteração "
                        "do volume físico. "

                        "Quando precisar de uma ferramenta, chame-a "
                        "diretamente. Não faça uma resposta provisória "
                        "antes da tool. "

                        "Depois da tool, responda usando o resultado. "

                        "Nunca diga que não possui internet se search_web "
                        "estiver disponível. "

                        "Nunca diga que não consegue ver se look estiver "
                        "disponível. "

                        "Luciano pode interromper você a qualquer momento. "
                        "Quando ele começar a falar, pare sua resposta "
                        "e escute a nova fala."
                    ),


                    "tools": TOOLS,

                    "tool_choice": "auto",


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
                                    "Assistente: AURI. "
                                    "Unitree é fabricante de robôs. "
                                    "Modelos incluem R1 e G1."
                                ),
                            },


                            "turn_detection": {

                                "type": "server_vad",

                                # Mantemos razoavelmente alto
                                # para reduzir eco do próprio speaker.
                                "threshold": 0.62,

                                "prefix_padding_ms": 300,

                                "silence_duration_ms": 700,

                                "create_response": True,

                                # PRINCIPAL CORREÇÃO DA 5.4.1:
                                # servidor cancela automaticamente
                                # resposta quando detectar nova fala.
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
                "🟢 AURI v0.5.4.1 ONLINE"
            )
            print(
                "===================================="
            )
            print("")
            print(
                "🌐 Real Web Tool"
            )
            print(
                "👁️ Real Vision Tool"
            )
            print(
                "🔊 Real Volume Tool"
            )
            print(
                "✋ Automatic Barge-in"
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


except Exception as error:

    print("")
    print("")
    print(
        "❌ AURI encerrada por erro:"
    )
    print(
        repr(error)
    )
