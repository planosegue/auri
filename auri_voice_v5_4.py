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
# AURI v0.5.4
# Realtime + Real Tools + Barge-in
# ============================================================

REALTIME_MODEL = "gpt-realtime-2.1-mini"
VISION_MODEL = "gpt-5.6-luna"
WEB_MODEL = "gpt-5.6-luna"

OPENAI_RATE = 24000

MIN_VOLUME = 10
MAX_VOLUME = 100
VOLUME_STEP = 15

# Evita o ponto exato de 0° das antenas.
ANTENNA_REST_OFFSET = 0.17

# Pequena proteção depois do fim real do speaker.
POST_SPEECH_GUARD = 0.45

# Evita interpretar um estalo/eco exatamente no começo
# da própria fala como interrupção.
BARGE_IN_ARM_DELAY = 0.60


# ============================================================
# ESTADO
# ============================================================

class AuriState:

    def __init__(self):

        self.assistant_speaking = False
        self.assistant_started_at = None

        self.vision_processing = False
        self.web_processing = False

        self.ignore_speech_until = 0.0

        self.audio_samples = 0
        self.audio_started_at = None

        self.current_volume = 80
        self.current_expression = None

        self.pending_tool_followup = False

        self.user_speaking = False


state = AuriState()


# ============================================================
# EXPRESSÕES
# ============================================================

EXPRESSIONS = {

    "neutral": {
        "head": create_head_pose(),
        "antennas": [
            ANTENNA_REST_OFFSET,
            -ANTENNA_REST_OFFSET
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
            -0.30
        ],
        "duration": 0.4,
    },

    "thinking": {
        "head": create_head_pose(
            yaw=-7,
            roll=7,
            degrees=True,
        ),
        "antennas": [
            0.12,
            -0.12
        ],
        "duration": 0.5,
    },

    "curious": {
        "head": create_head_pose(
            yaw=7,
            roll=-8,
            degrees=True,
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
            degrees=True,
        ),
        "antennas": [
            0.38,
            -0.38
        ],
        "duration": 0.35,
    },
}


def expression(mini, name):

    # Não reaplica desnecessariamente
    # a mesma posição aos servos.
    if state.current_expression == name:
        return

    state.current_expression = name

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

    # Reachy: stereo 16 kHz.
    # Canal 0 foi o caminho validado.
    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(
        samples,
        -1.0,
        1.0
    )

    # 16k -> 24k
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

    # 24k -> 16k
    return resample_poly(
        samples_24k,
        2,
        3
    ).astype(np.float32)


# ============================================================
# PLAYER / BARGE-IN
# ============================================================

def clear_reachy_player(mini):

    """
    Compatibilidade entre versões do SDK.
    Algumas expõem clear_player no media,
    outras no backend media.audio.
    """

    try:

        if hasattr(
            mini.media,
            "clear_player"
        ):

            mini.media.clear_player()

            print(
                "🧹 Speaker buffer limpo"
            )

            return True


        audio_backend = getattr(
            mini.media,
            "audio",
            None
        )

        if (
            audio_backend is not None
            and hasattr(
                audio_backend,
                "clear_player"
            )
        ):

            audio_backend.clear_player()

            print(
                "🧹 Speaker buffer limpo"
            )

            return True


        print(
            "⚠️ clear_player não disponível "
            "nesta versão do SDK"
        )

        return False


    except Exception as error:

        print(
            "⚠️ Falha ao limpar player:",
            repr(error)
        )

        return False


def reset_playback_state():

    state.assistant_speaking = False
    state.assistant_started_at = None

    state.audio_samples = 0
    state.audio_started_at = None


async def interrupt_assistant(
    mini,
    connection
):

    if not state.assistant_speaking:
        return False


    if state.assistant_started_at is not None:

        elapsed = (
            time.monotonic()
            - state.assistant_started_at
        )

        if elapsed < BARGE_IN_ARM_DELAY:

            print(
                "\n🛡️ Barge-in ignorado "
                "(início da própria fala)"
            )

            return False


    print("")
    print("==============================")
    print("✋ INTERRUPÇÃO DETECTADA")
    print("==============================")


    # Primeiro para a geração no servidor.
    try:

        await connection.response.cancel()

    except Exception as error:

        print(
            "⚠️ Cancel response:",
            repr(error)
        )


    # Depois elimina o áudio que já estava
    # fisicamente enfileirado.
    clear_reachy_player(
        mini
    )


    reset_playback_state()


    # O usuário já está falando.
    expression(
        mini,
        "listening"
    )


    print(
        "🎙️ AURI interrompida. Ouvindo Luciano..."
    )


    return True


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
# WEB SEARCH TOOL
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
                "Priorize fontes oficiais, fabricantes, "
                "documentação técnica e informações recentes. "
                "Responda em português brasileiro. "
                "Seja factual e relativamente conciso. "
                "Não invente dados.\n\n"
                f"Pesquisa: {query}"
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
# VISION TOOL
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


    # Aguarda movimento antes do frame.
    await asyncio.sleep(
        0.55
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


    # BGR -> RGB
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

            max_output_tokens=150,

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",

                            "text": (
                                "Você é a percepção visual da AURI. "
                                "Analise somente o que realmente está "
                                "visível na imagem. "
                                "Responda em português brasileiro. "
                                "Não invente detalhes. "
                                "Produza uma percepção objetiva e curta "
                                "para que AURI possa responder "
                                "conversacionalmente.\n\n"
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
# EXECUTOR DE REAL TOOLS
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


        # Durante visão/web, não enviamos ruído
        # do movimento/processamento.
        #
        # IMPORTANTE:
        # durante assistant_speaking o microfone
        # CONTINUA indo para o Realtime para
        # permitir barge-in.

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
# SPEAKER SYNC
# ============================================================

async def wait_for_speaker():

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
            f"\n🔊 Aguardando speaker: "
            f"{remaining:.2f}s"
        )


        await asyncio.sleep(
            remaining
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


            state.user_speaking = True


            # BARGE-IN
            if state.assistant_speaking:

                interrupted = await interrupt_assistant(
                    mini,
                    connection
                )


                if interrupted:
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


            state.user_speaking = False


            expression(
                mini,
                "thinking"
            )


            print(
                "\n🧠 Processando..."
            )


        # ------------------------------------------------
        # TRANSCRIÇÃO DO USUÁRIO
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


            # Não criamos response aqui.
            # Com server VAD configurado abaixo,
            # create_response=True cuidará do turno.


        # ------------------------------------------------
        # REAL TOOL CALL
        # ------------------------------------------------

        elif event.type == \
            "response.function_call_arguments.done":


            print("")
            print(
                f"🎯 TOOL escolhida: {event.name}"
            )


            result = await execute_tool(
                client,
                mini,
                event.name,
                event.arguments
            )


            # Entrega o resultado para o mesmo
            # call_id que o modelo criou.
            await connection.conversation.item.create(

                item={
                    "type": "function_call_output",

                    "call_id": event.call_id,

                    "output": json.dumps(
                        result,
                        ensure_ascii=False
                    ),
                }
            )


            # Não criamos a próxima resposta ainda.
            # Esperamos o response.done da resposta
            # que originou a function call.
            state.pending_tool_followup = True


            expression(
                mini,
                "thinking"
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
        # TEXTO CORRESPONDENTE À VOZ
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
            # CANCELADO POR BARGE-IN
            # --------------------------------------------

            if status == "cancelled":

                print("")
                print(
                    "✋ Resposta anterior cancelada"
                )


                reset_playback_state()


                # Usuário ainda pode estar falando.
                if state.user_speaking:

                    expression(
                        mini,
                        "listening"
                    )


                continue


            # --------------------------------------------
            # TOOL FOI EXECUTADA
            # --------------------------------------------

            if state.pending_tool_followup:

                state.pending_tool_followup = False


                print("")
                print(
                    "🛠️ Resultado da tool entregue à AURI"
                )


                # Agora gera a resposta final baseada
                # no function_call_output.
                await connection.response.create()


                continue


            # --------------------------------------------
            # RESPOSTA FALADA NORMAL
            # --------------------------------------------

            if state.assistant_speaking:

                print("")


                await wait_for_speaker()


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
# TOOLS
# ============================================================

TOOLS = [

    {
        "type": "function",

        "name": "search_web",

        "description": (
            "Pesquisa a internet quando a pergunta depende "
            "de informações atuais, recentes, notícias, preços, "
            "lançamentos, documentação online ou quando o usuário "
            "pede explicitamente uma pesquisa. "
            "Use esta ferramenta em vez de alegar que não possui "
            "acesso à internet."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "query": {
                    "type": "string",
                    "description": (
                        "Consulta completa e clara para pesquisar."
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
            "Olha através da câmera física do corpo da AURI. "
            "Use para perguntas sobre o ambiente atual, objetos, "
            "pessoas, texto visível, cores ou qualquer coisa "
            "que exija enxergar."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "question": {
                    "type": "string",
                    "description": (
                        "O que deve ser observado ou respondido "
                        "usando a visão."
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
            "Controla o volume físico do speaker da AURI. "
            "Para 'mais alto', use action='up'. "
            "Para 'mais baixo', use action='down'. "
            "Para um valor específico, use action='set' "
            "e informe percent."
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
        "🤖 AURI v0.5.4"
    )
    print(
        "   Real Tools + Barge-in"
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

                        "Não reinicie conversas nem repita constantemente "
                        "'como posso te ajudar hoje?'. "

                        "Prefira respostas naturais e concisas, normalmente "
                        "de uma a três frases, salvo quando Luciano pedir "
                        "mais detalhes. "

                        "Você possui ferramentas reais. "

                        "Se precisar de informação atual ou se Luciano pedir "
                        "pesquisa, use search_web. "

                        "Se precisar enxergar o ambiente, use look. "

                        "Se Luciano pedir alteração de volume, use set_volume. "

                        "Quando precisar de uma ferramenta, CHAME A TOOL "
                        "IMEDIATAMENTE. Não explique antes que vai usá-la "
                        "e não dê uma resposta provisória. "

                        "Depois que a ferramenta retornar, responda usando "
                        "o resultado. "

                        "Nunca diga que não possui acesso à internet quando "
                        "search_web estiver disponível. "

                        "Nunca diga que não consegue ver quando look estiver "
                        "disponível. "

                        "Você pode ser interrompida enquanto fala. "
                        "Se Luciano começar a falar, pare e escute."
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

                                # Um pouco mais conservador
                                # por causa do próprio speaker.
                                "threshold": 0.62,

                                "prefix_padding_ms": 300,

                                "silence_duration_ms": 700,

                                # Gera o primeiro response
                                # automaticamente ao fim da fala.
                                "create_response": True,

                                # Barge-in é controlado por nós:
                                # response.cancel + clear_player.
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
                "🟢 AURI v0.5.4 ONLINE"
            )
            print(
                "===================================="
            )
            print("")
            print(
                "Real Tools:"
            )
            print(
                "  🌐 search_web"
            )
            print(
                "  👁️ look"
            )
            print(
                "  🔊 set_volume"
            )
            print("")
            print(
                "Barge-in:"
            )
            print(
                "  Interrompa AURI enquanto ela fala."
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
