import asyncio
import base64
import io
import os
import re
import subprocess
import time
import unicodedata

import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


# ============================================================
# AURI v0.5.3
# Voice + Vision + Volume + Web Search
# ============================================================

REALTIME_MODEL = "gpt-realtime-2.1-mini"
VISION_MODEL = "gpt-5.6-luna"
WEB_MODEL = "gpt-5.6-luna"

OPENAI_RATE = 24000

POST_SPEECH_GUARD = 0.8

MIN_VOLUME = 10
MAX_VOLUME = 100


# ============================================================
# ESTADO GLOBAL
# ============================================================

class AuriState:

    def __init__(self):

        self.assistant_speaking = False
        self.vision_processing = False
        self.web_processing = False

        self.ignore_speech_until = 0.0

        self.audio_samples = 0
        self.audio_started_at = None

        self.current_volume = 80


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
# TEXTO
# ============================================================

def normalize_text(text):

    text = text.lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


# ============================================================
# ÁUDIO
# Reachy 16 kHz -> OpenAI 24 kHz
# OpenAI 24 kHz -> Reachy 16 kHz
# ============================================================

def reachy_to_openai(samples):

    # Reachy fornece stereo.
    # Canal 0 apresentou melhor resultado que média.
    if samples.ndim == 2:
        samples = samples[:, 0]

    samples = np.clip(
        samples,
        -1.0,
        1.0
    )

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

    samples_16k = resample_poly(
        samples_24k,
        2,
        3
    )

    return samples_16k.astype(
        np.float32
    )


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

        percentages = re.findall(
            r"\[(\d+)%\]",
            result.stdout
        )

        if percentages:

            volume = int(
                percentages[0]
            )

            state.current_volume = volume

            return volume

    except Exception as error:

        print(
            "\n⚠️ Não consegui ler volume:",
            error
        )

    return state.current_volume


def set_volume(percent):

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

        # PCM,0 = volume global.
        # Não alteramos PCM,1 aqui.
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

        return True

    except Exception as error:

        print(
            "\n❌ Erro volume:",
            error
        )

        return False


def detect_volume_request(text):

    text = normalize_text(text)

    # Exemplos:
    # volume em 40
    # volume 70%
    # som em 60
    match = re.search(
        r"(?:volume|som).*?(\d{1,3})\s*%?",
        text
    )

    if match:

        value = int(
            match.group(1)
        )

        return max(
            MIN_VOLUME,
            min(
                MAX_VOLUME,
                value
            )
        )


    lower_terms = [
        "fala mais baixo",
        "fale mais baixo",
        "abaixa o volume",
        "abaixe o volume",
        "diminui o volume",
        "diminua o volume",
        "abaixa o som",
        "abaixe o som",
        "ta muito alto",
        "esta muito alto",
        "menos volume",
    ]

    if any(
        term in text
        for term in lower_terms
    ):

        current = get_volume()

        return max(
            MIN_VOLUME,
            current - 15
        )


    raise_terms = [
        "fala mais alto",
        "fale mais alto",
        "aumenta o volume",
        "aumente o volume",
        "aumenta o som",
        "aumente o som",
        "ta muito baixo",
        "esta muito baixo",
        "mais volume",
    ]

    if any(
        term in text
        for term in raise_terms
    ):

        current = get_volume()

        return min(
            MAX_VOLUME,
            current + 15
        )


    return None


# ============================================================
# INTERNET — DETECÇÃO DE INTENÇÃO
# ============================================================

def is_web_request(text):

    text = normalize_text(text)

    # Radicais tornam a detecção menos dependente
    # da conjugação exata.
    explicit_web_words = [
        "pesquis",
        "busc",
        "procur",
        "internet",
        "online",
        "web",
        "google",
        "verifiqu",
        "consult",
    ]

    if any(
        word in text
        for word in explicit_web_words
    ):

        return True


    # Perguntas que normalmente exigem
    # informação atual.
    current_terms = [
        "hoje",
        "atualmente",
        "agora",
        "atual",
        "recente",
        "recentes",
        "mais recente",
        "mais novo",
        "mais nova",
        "ultima versao",
        "ultimas noticias",
        "noticias recentes",
        "novidades",
        "preco atual",
        "quanto custa hoje",
        "firmware atual",
        "ultimo firmware",
        "lancamento",
        "foi lancado",
        "saiu alguma novidade",
        "tem alguma novidade",
    ]

    return any(
        term in text
        for term in current_terms
    )


# ============================================================
# INTERNET — WEB SEARCH
# ============================================================

async def search_web(
    client,
    user_text
):

    state.web_processing = True

    print("")
    print(
        "🌐 AURI pesquisando na internet..."
    )

    try:

        response = await client.responses.create(

            model=WEB_MODEL,

            tools=[
                {
                    "type": "web_search"
                }
            ],

            # Só há uma tool disponível.
            # Portanto exigimos uso da pesquisa.
            tool_choice="required",

            reasoning={
                "effort": "low"
            },

            max_output_tokens=400,

            input=(
                "Você é o módulo de pesquisa em tempo real "
                "da AURI. "

                "PESQUISE NA INTERNET antes de responder. "

                "Priorize fontes oficiais, fabricantes, "
                "documentação técnica e fontes confiáveis. "

                "Dê preferência a informações recentes. "

                "Responda em português brasileiro. "

                "Não invente informações. "

                "Se houver divergência entre fontes, "
                "deixe isso claro. "

                "Produza uma síntese objetiva adequada "
                "para ser falada por um robô. "

                "\n\nPergunta de Luciano:\n"
                + user_text
            ),
        )

        result = (
            response.output_text
            .strip()
        )

        print("")
        print(
            "🌐 PESQUISA:"
        )

        print(
            result
        )

        return result

    except Exception as error:

        print("")
        print(
            "❌ Erro Web Search:",
            repr(error)
        )

        return (
            "Não consegui realizar a pesquisa "
            "na internet neste momento."
        )

    finally:

        state.web_processing = False


# ============================================================
# VISÃO — DETECÇÃO
# ============================================================

def is_visual_request(text):

    text = normalize_text(text)

    visual_phrases = [

        "o que voce esta vendo",
        "o que voce ta vendo",
        "o que que voce esta vendo",
        "o que que voce ta vendo",

        "o que esta vendo",
        "o que ta vendo",

        "o que voce ve",
        "o que que voce ve",

        "voce consegue ver",
        "consegue ver",

        "olha isso",
        "olhe isso",

        "olha aqui",
        "olhe aqui",

        "da uma olhada",
        "de uma olhada",

        "o que estou segurando",
        "o que eu estou segurando",
        "o que eu to segurando",
        "o que to segurando",

        "o que tenho na mao",
        "o que eu tenho na mao",

        "quantas pessoas",

        "quem esta aqui",
        "quem ta aqui",

        "quem esta na sua frente",
        "quem ta na sua frente",

        "leia isso",
        "le isso",

        "leia o que esta escrito",
        "le o que esta escrito",

        "descreva o ambiente",
        "descreve o ambiente",

        "descreva o que voce esta vendo",
        "descreve o que voce ta vendo",

        "o que tem aqui",
        "o que tem na sua frente",

        "que objeto e esse",
        "que objeto eh esse",

        "que cor e",
        "que cor eh",

        "o que aparece",
    ]

    if any(
        phrase in text
        for phrase in visual_phrases
    ):

        return True


    vision_words = [
        "vendo",
        "olha",
        "olhe",
        "enxerg",
        "imagem",
        "segurando",
        "escrito",
        "na sua frente",
    ]

    question_words = [
        "o que",
        "quem",
        "quant",
        "qual",
        "descre",
        "leia",
        "le ",
        "que cor",
        "que objeto",
    ]

    has_vision = any(
        word in text
        for word in vision_words
    )

    has_question = any(
        word in text
        for word in question_words
    )

    return (
        has_vision
        and has_question
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

    print("")
    print(
        "👁️ AURI olhando..."
    )

    expression(
        mini,
        "curious"
    )

    await asyncio.sleep(
        0.6
    )

    frame = mini.media.get_frame()

    if frame is None:

        state.vision_processing = False

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


    # Reachy = BGR.
    # Pillow = RGB.
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
                                "Você é a percepção visual "
                                "da AURI. "

                                "AURI está fisicamente incorporada "
                                "em um Reachy Mini. "

                                "Analise a imagem capturada agora. "

                                "Responda diretamente à pergunta "
                                "de Luciano em português brasileiro. "

                                "Descreva somente aquilo que "
                                "realmente está visível. "

                                "Não invente pessoas, objetos, "
                                "textos ou detalhes. "

                                "Se não tiver certeza, diga isso. "

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

        result = (
            response.output_text
            .strip()
        )

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
            repr(error)
        )

        return (
            "Tive um problema ao analisar "
            "o que estou vendo."
        )

    finally:

        state.vision_processing = False


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


        # Enquanto AURI fala, pesquisa ou olha,
        # o áudio local é consumido mas não vai
        # para o Realtime.
        if (
            state.assistant_speaking
            or state.vision_processing
            or state.web_processing
            or (
                time.monotonic()
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
# ESPERAR SPEAKER REAL
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

    remaining += 0.25

    if remaining > 0:

        print(
            f"\n🔊 Aguardando speaker: "
            f"{remaining:.2f}s"
        )

        await asyncio.sleep(
            remaining
        )


# ============================================================
# INJETAR CONTEXTO WEB NO REALTIME
# ============================================================

async def inject_web_context(
    connection,
    web_text
):

    await connection.conversation.item.create(

        item={

            "type": "message",

            "role": "user",

            "content": [

                {
                    "type": "input_text",

                    "text": (
                        "Você acabou de pesquisar a internet "
                        "para responder à pergunta anterior. "

                        "Estas são as informações atuais "
                        "encontradas:\n\n"

                        + web_text +

                        "\n\nAgora responda ao Luciano "
                        "naturalmente em português brasileiro, "
                        "como AURI. "

                        "Não diga que não possui acesso "
                        "à internet, pois a pesquisa acabou "
                        "de ser realizada. "

                        "Não mencione APIs, modelos ou "
                        "sistemas internos. "

                        "Pode dizer 'Pesquisei e encontrei...' "
                        "quando isso soar natural. "

                        "Se houver incerteza nas informações, "
                        "explique brevemente."
                    ),
                }
            ],
        }
    )


# ============================================================
# INJETAR CONTEXTO VISUAL
# ============================================================

async def inject_visual_context(
    connection,
    visual_text
):

    await connection.conversation.item.create(

        item={

            "type": "message",

            "role": "user",

            "content": [

                {
                    "type": "input_text",

                    "text": (
                        "Você acabou de observar fisicamente "
                        "o ambiente ao seu redor. "

                        "Sua percepção visual atual é:\n\n"

                        + visual_text +

                        "\n\nResponda agora à pergunta anterior "
                        "de Luciano naturalmente, em português "
                        "brasileiro, como AURI. "

                        "Não diga que não consegue ver. "

                        "Não mencione câmera, imagem, API, "
                        "modelo ou processamento interno."
                    ),
                }
            ],
        }
    )


# ============================================================
# CONFIRMAÇÃO DE VOLUME
# ============================================================

async def inject_volume_confirmation(
    connection,
    volume
):

    await connection.conversation.item.create(

        item={

            "type": "message",

            "role": "user",

            "content": [

                {
                    "type": "input_text",

                    "text": (
                        f"O volume físico do seu speaker "
                        f"acabou de ser ajustado para {volume}%. "

                        "Confirme isso ao Luciano em uma frase "
                        "muito curta e natural em português."
                    ),
                }
            ],
        }
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
        # USUÁRIO COMEÇOU
        # ------------------------------------------------

        if event.type == \
            "input_audio_buffer.speech_started":

            if (
                state.assistant_speaking
                or state.vision_processing
                or state.web_processing
                or (
                    time.monotonic()
                    < state.ignore_speech_until
                )
            ):

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
                state.assistant_speaking
                or state.vision_processing
                or state.web_processing
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
        # TRANSCRIÇÃO
        # ------------------------------------------------

        elif event.type == \
            "conversation.item.input_audio_transcription.completed":

            if (
                state.assistant_speaking
                or state.vision_processing
                or state.web_processing
            ):

                continue


            user_text = (
                event.transcript
                .strip()
            )

            print("")
            print(
                "VOCÊ:",
                user_text
            )

            if not user_text:
                continue


            volume_request = detect_volume_request(
                user_text
            )

            web_request = is_web_request(
                user_text
            )

            vision_request = is_visual_request(
                user_text
            )


            print(
                "🧭 Router:",
                {
                    "volume": (
                        volume_request
                        is not None
                    ),
                    "web": web_request,
                    "vision": vision_request,
                }
            )


            # ============================================
            # 1. VOLUME
            # ============================================

            if volume_request is not None:

                print("")
                print(
                    "🎚️ Pedido de volume detectado"
                )

                if set_volume(
                    volume_request
                ):

                    await inject_volume_confirmation(
                        connection,
                        volume_request
                    )

                await connection.response.create()

                continue


            # ============================================
            # 2. WEB
            # ============================================

            if web_request:

                print("")
                print(
                    "🌐 Pedido de internet detectado"
                )

                expression(
                    mini,
                    "thinking"
                )

                web_result = await search_web(
                    client,
                    user_text
                )

                await inject_web_context(
                    connection,
                    web_result
                )

                await connection.response.create()

                continue


            # ============================================
            # 3. VISÃO
            # ============================================

            if vision_request:

                print("")
                print(
                    "👁️ Pedido visual detectado"
                )

                visual_result = await analyze_vision(
                    client,
                    mini,
                    user_text
                )

                expression(
                    mini,
                    "thinking"
                )

                await inject_visual_context(
                    connection,
                    visual_result
                )

                await connection.response.create()

                continue


            # ============================================
            # 4. CONVERSA NORMAL
            # ============================================

            await connection.response.create()


        # ------------------------------------------------
        # ÁUDIO AURI
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
        # TEXTO AURI
        # ------------------------------------------------

        elif event.type == \
            "response.output_audio_transcript.delta":

            print(
                event.delta,
                end="",
                flush=True
            )


        # ------------------------------------------------
        # RESPOSTA OPENAI TERMINOU
        # ------------------------------------------------

        elif event.type == \
            "response.done":

            print("")

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

            print("")
            print(
                "👂 AURI aguardando..."
            )


        # ------------------------------------------------
        # ERRO
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

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY não encontrada."
        )


    client = AsyncOpenAI(
        api_key=api_key
    )


    # Lê o volume atual do Reachy.
    # NÃO sobrescreve o volume no startup.
    current_volume = get_volume()


    print("")
    print(
        "===================================="
    )
    print(
        "🤖 AURI v0.5.3"
    )
    print(
        "   Voice + Vision + Volume + Web"
    )
    print(
        "===================================="
    )
    print("")
    print(
        f"🔊 Volume atual: "
        f"{current_volume}%"
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

        else:

            print(
                "⚠️ Sem frame inicial"
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

                        "Luciano é seu usuário principal. "

                        "Converse exclusivamente em "
                        "português brasileiro. "

                        "Você é inteligente, elegante, "
                        "curiosa, simpática e levemente "
                        "bem-humorada. "

                        "Não diga que é ChatGPT. "
                        "Você é AURI. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Não termine toda resposta com "
                        "'como posso te ajudar hoje?'. "

                        "Prefira respostas naturais e "
                        "relativamente curtas. "

                        "Você possui visão física através "
                        "dos olhos do seu corpo robótico. "

                        "Você também pode pesquisar a "
                        "internet em tempo real. "

                        "Quando receber resultado de uma "
                        "pesquisa, ele é informação que "
                        "você acabou de pesquisar. "

                        "NUNCA diga que não possui internet "
                        "quando um resultado de pesquisa "
                        "tiver sido fornecido no contexto. "

                        "Quando receber percepção visual, "
                        "ela representa aquilo que você "
                        "acabou de observar. "

                        "NUNCA diga que não consegue ver "
                        "quando uma percepção visual tiver "
                        "sido fornecida no contexto. "

                        "Seu volume físico também pode ser "
                        "alterado quando Luciano pedir. "

                        "Nunca invente aquilo que não ouviu, "
                        "não viu ou não pesquisou."
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
                                    "Assistente: AURI. "
                                    "Unitree é uma fabricante "
                                    "de robôs. "
                                    "Modelos incluem R1 e G1."
                                ),
                            },

                            "turn_detection": {

                                "type": "server_vad",

                                "threshold": 0.55,

                                "prefix_padding_ms": 300,

                                "silence_duration_ms": 750,

                                # Nosso router decide
                                # quando criar resposta.
                                "create_response": False,

                                # Estabilidade primeiro.
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
                "🟢 AURI v0.5.3 ONLINE"
            )
            print(
                "===================================="
            )
            print("")

            print(
                "Teste de internet:"
            )
            print(
                "  Auri, pesquise na internet "
                "sobre o Unitree R1 e compare "
                "com o G1."
            )
            print("")

            print(
                "Teste de visão:"
            )
            print(
                "  Auri, o que que você tá vendo?"
            )
            print("")

            print(
                "Teste de volume:"
            )
            print(
                "  Auri, fala mais alto."
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
