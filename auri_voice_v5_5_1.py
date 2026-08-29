import asyncio
import base64
import io
import json
import os
import re
import sqlite3
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

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
# MEMORY
# ============================================================

MEMORY_DB = Path(
    "/home/pollen/auri/data/auri_memory.db"
)

MEMORY_MAX_RESULTS = 8
MEMORY_MAX_FACT_CHARS = 800


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
        self.memory_processing = False

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
# MEMORY ENGINE
# ============================================================

def normalize_memory_text(text):

    text = (text or "").lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def memory_connect():

    MEMORY_DB.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        MEMORY_DB
    )

    conn.row_factory = sqlite3.Row

    return conn


def initialize_memory():

    MEMORY_DB.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with memory_connect() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                importance INTEGER NOT NULL DEFAULT 5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(memories)"
            ).fetchall()
        }

        if "normalized_fact" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN normalized_fact TEXT
                """
            )

        if "access_count" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN access_count
                INTEGER NOT NULL DEFAULT 0
                """
            )

        if "last_accessed_at" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN last_accessed_at TEXT
                """
            )

        rows = conn.execute(
            """
            SELECT id, fact
            FROM memories
            WHERE normalized_fact IS NULL
               OR normalized_fact = ''
            """
        ).fetchall()

        for row in rows:

            conn.execute(
                """
                UPDATE memories
                SET normalized_fact = ?
                WHERE id = ?
                """,
                (
                    normalize_memory_text(
                        row["fact"]
                    ),
                    row["id"],
                )
            )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memory_normalized_fact
            ON memories(normalized_fact)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memory_subject
            ON memories(subject)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memory_category
            ON memories(category)
            """
        )

        conn.commit()


def memory_contains_secret(text):

    normalized = normalize_memory_text(
        text
    )

    patterns = [
        r"sk-[a-z0-9_-]{12,}",
        r"api[_ -]?key",
        r"password",
        r"senha",
        r"secret",
        r"token",
        r"bearer ",
        r"private[_ -]?key",
    ]

    return any(
        re.search(
            pattern,
            normalized,
            re.IGNORECASE
        )
        for pattern in patterns
    )


def remember_memory(args):

    state.memory_processing = True

    try:

        subject = (
            args.get("subject")
            or "general"
        ).strip()

        fact = (
            args.get("fact")
            or ""
        ).strip()

        category = (
            args.get("category")
            or "general"
        ).strip()

        importance = int(
            args.get(
                "importance",
                5
            )
        )

        importance = max(
            1,
            min(
                10,
                importance
            )
        )

        if not fact:

            return {
                "ok": False,
                "error": "Memória vazia.",
            }

        if len(fact) > MEMORY_MAX_FACT_CHARS:

            return {
                "ok": False,
                "error": (
                    "Memória longa demais. "
                    "Resuma antes de salvar."
                ),
            }

        if memory_contains_secret(fact):

            return {
                "ok": False,
                "error": (
                    "Memória recusada por conter "
                    "possível senha, token ou segredo."
                ),
            }

        normalized = normalize_memory_text(
            fact
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with memory_connect() as conn:

            existing = conn.execute(
                """
                SELECT
                    id,
                    fact,
                    importance
                FROM memories
                WHERE normalized_fact = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    normalized,
                )
            ).fetchone()

            # --------------------------------------------
            # DEDUPLICAÇÃO
            # --------------------------------------------

            if existing:

                new_importance = max(
                    int(existing["importance"]),
                    importance
                )

                conn.execute(
                    """
                    UPDATE memories
                    SET
                        subject = ?,
                        category = ?,
                        importance = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        subject,
                        category,
                        new_importance,
                        now,
                        existing["id"],
                    )
                )

                conn.commit()

                print("")
                print(
                    "🧠 Memória já existente — "
                    "registro atualizado"
                )

                return {
                    "ok": True,
                    "deduplicated": True,
                    "memory_id": existing["id"],
                    "fact": existing["fact"],
                }

            cursor = conn.execute(
                """
                INSERT INTO memories (
                    subject,
                    fact,
                    category,
                    importance,
                    created_at,
                    updated_at,
                    normalized_fact,
                    access_count,
                    last_accessed_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, 0, NULL
                )
                """,
                (
                    subject,
                    fact,
                    category,
                    importance,
                    now,
                    now,
                    normalized,
                )
            )

            conn.commit()

            memory_id = cursor.lastrowid

        print("")
        print(
            f"🧠 MEMORY #{memory_id}"
        )
        print(
            f"   {subject} → {fact}"
        )

        return {
            "ok": True,
            "deduplicated": False,
            "memory_id": memory_id,
            "fact": fact,
        }

    except Exception as error:

        print(
            "\n❌ Memory remember:",
            repr(error)
        )

        return {
            "ok": False,
            "error": str(error),
        }

    finally:

        state.memory_processing = False


MEMORY_STOPWORDS = {
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "e",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "que",
    "qual",
    "quais",
    "quem",
    "como",
    "por",
    "para",
    "com",
    "sobre",
    "meu",
    "minha",
    "lembra",
    "lembrar",
}


def memory_tokens(query):

    normalized = normalize_memory_text(
        query
    )

    words = re.findall(
        r"[a-z0-9]+",
        normalized
    )

    return list(
        dict.fromkeys(
            word
            for word in words
            if (
                len(word) >= 3
                and word not in MEMORY_STOPWORDS
            )
        )
    )


def recall_memory(args):

    state.memory_processing = True

    try:

        query = (
            args.get("query")
            or ""
        ).strip()

        limit = int(
            args.get(
                "limit",
                MEMORY_MAX_RESULTS
            )
        )

        limit = max(
            1,
            min(
                MEMORY_MAX_RESULTS,
                limit
            )
        )

        tokens = memory_tokens(
            query
        )

        normalized_query = (
            normalize_memory_text(
                query
            )
        )

        with memory_connect() as conn:

            rows = conn.execute(
                """
                SELECT
                    id,
                    subject,
                    fact,
                    category,
                    importance,
                    updated_at,
                    access_count
                FROM memories
                ORDER BY
                    importance DESC,
                    updated_at DESC
                LIMIT 250
                """
            ).fetchall()

            scored = []

            for row in rows:

                subject_norm = (
                    normalize_memory_text(
                        row["subject"]
                    )
                )

                category_norm = (
                    normalize_memory_text(
                        row["category"]
                    )
                )

                fact_norm = (
                    normalize_memory_text(
                        row["fact"]
                    )
                )

                haystack = (
                    subject_norm
                    + " "
                    + category_norm
                    + " "
                    + fact_norm
                )

                score = 0.0

                if (
                    normalized_query
                    and normalized_query
                    in haystack
                ):

                    score += 12.0

                for token in tokens:

                    if token in fact_norm:
                        score += 3.0

                    if token in subject_norm:
                        score += 2.0

                    if token in category_norm:
                        score += 1.0

                score += (
                    int(row["importance"])
                    * 0.20
                )

                if score > 0:

                    scored.append(
                        (
                            score,
                            row
                        )
                    )

            scored.sort(
                key=lambda item: (
                    item[0],
                    item[1]["importance"],
                    item[1]["updated_at"],
                ),
                reverse=True
            )

            selected = scored[:limit]

            now = datetime.now(
                timezone.utc
            ).isoformat()

            memories = []

            for score, row in selected:

                conn.execute(
                    """
                    UPDATE memories
                    SET
                        access_count =
                            access_count + 1,
                        last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        row["id"],
                    )
                )

                memories.append(
                    {
                        "memory_id": row["id"],
                        "subject": row["subject"],
                        "fact": row["fact"],
                        "category": row["category"],
                        "importance": row["importance"],
                        "score": round(
                            score,
                            2
                        ),
                    }
                )

            conn.commit()

        print("")
        print(
            f"🔎 MEMORY → {query}"
        )

        if not memories:

            print(
                "   Nenhuma memória encontrada"
            )

        else:

            for memory in memories:

                print(
                    f"   #{memory['memory_id']} "
                    f"[{memory['category']}] "
                    f"{memory['fact']}"
                )

        return {
            "ok": True,
            "query": query,
            "count": len(memories),
            "memories": memories,
        }

    except Exception as error:

        print(
            "\n❌ Memory recall:",
            repr(error)
        )

        return {
            "ok": False,
            "error": str(error),
        }

    finally:

        state.memory_processing = False


def memory_stats():

    try:

        with memory_connect() as conn:

            count = conn.execute(
                """
                SELECT COUNT(*)
                FROM memories
                """
            ).fetchone()[0]

        return int(count)

    except Exception:

        return 0


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


    if name == "remember":

        return remember_memory(
            args
        )


    if name == "recall":

        return recall_memory(
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
            or state.memory_processing
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
# MEMORY REAL TOOLS
# ============================================================

TOOLS.extend(
    [

        {
            "type": "function",

            "name": "remember",

            "description": (
                "Salva uma memória persistente entre sessões. "
                "Use quando Luciano pedir explicitamente para "
                "lembrar, guardar ou não esquecer, ou quando "
                "uma decisão ou fato claramente duradouro do "
                "projeto mereça persistir. "
                "Não memorize conversa casual, senhas, API keys, "
                "tokens, credenciais ou outros segredos."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "subject": {
                        "type": "string",
                        "description": (
                            "Assunto principal, por exemplo "
                            "Luciano, AURI, AURI Physical ou Unitree."
                        ),
                    },

                    "fact": {
                        "type": "string",
                        "description": (
                            "Fato autocontido e duradouro."
                        ),
                    },

                    "category": {
                        "type": "string",
                        "enum": [
                            "identity",
                            "project",
                            "preference",
                            "decision",
                            "technical",
                            "relationship",
                            "general"
                        ],
                    },

                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },

                "required": [
                    "subject",
                    "fact",
                    "category",
                    "importance"
                ],

                "additionalProperties": False,
            },
        },


        {
            "type": "function",

            "name": "recall",

            "description": (
                "Consulta a memória persistente da AURI. "
                "Use quando Luciano perguntar sobre algo "
                "ensinado, decidido ou discutido em sessões "
                "anteriores, ou perguntar se você lembra. "
                "Se nenhuma memória relevante existir, "
                "não invente uma lembrança."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "Descrição curta do que procurar."
                        ),
                    },

                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                    },
                },

                "required": [
                    "query"
                ],

                "additionalProperties": False,
            },
        },

    ]
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


    initialize_memory()


    client = AsyncOpenAI(
        api_key=api_key
    )


    current_volume = get_volume()
    memories_count = memory_stats()


    print("")
    print(
        "===================================="
    )
    print(
        "🤖 AURI v0.5.5.1"
    )
    print(
        " Refined Voice + Persistent Memory"
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

    print(
        f"🧠 Memórias persistentes: "
        f"{memories_count}"
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

                        "Use português brasileiro neutro, natural "
                        "e consistente. "

                        "Use pronúncia brasileira clara e evite "
                        "qualquer sotaque estrangeiro perceptível. "

                        "Mantenha ritmo conversacional calmo, natural "
                        "e consistente entre respostas. "

                        "Evite exagero de entusiasmo, teatralidade "
                        "ou mudanças desnecessárias de prosódia. "

                        "Seja inteligente, elegante, curiosa, simpática "
                        "e levemente bem-humorada. "

                        "Evite gírias, vícios de linguagem e informalidade "
                        "excessiva. "

                        "Não diga que é ChatGPT. Você é AURI. "

                        "Mantenha o contexto da conversa. "

                        "Não reinicie a conversa sem motivo. "

                        "Não repita constantemente frases como "
                        "'como posso te ajudar hoje?'. "

                        "Para perguntas simples, responda de forma curta "
                        "e direta, normalmente em uma ou duas frases. "

                        "Só aprofunde quando Luciano pedir detalhes "
                        "ou quando o assunto realmente exigir. "

                        "Não ofereça ajuda adicional ao final de toda "
                        "resposta se isso não for necessário. "

                        "Você possui ferramentas reais. "

                        "Use search_web somente quando a resposta depender "
                        "de informação atual, recente, online, ou quando "
                        "Luciano pedir explicitamente uma pesquisa. "

                        "Não use internet desnecessariamente para fatos "
                        "históricos ou conhecimento estável. "

                        "Use look quando precisar enxergar algo no ambiente. "

                        "Use set_volume quando Luciano pedir alteração "
                        "do volume físico. "

                        "Você possui memória persistente entre sessões. "

                        "Use remember quando Luciano pedir explicitamente "
                        "para lembrar, guardar ou não esquecer um fato, "
                        "preferência ou decisão duradoura. "

                        "Você também pode usar remember para uma decisão "
                        "claramente importante e duradoura do projeto. "

                        "Não memorize conversa casual ou informação temporária. "

                        "Nunca memorize senhas, API keys, tokens, credenciais "
                        "ou qualquer segredo. "

                        "Use recall quando Luciano perguntar sobre algo "
                        "que pode ter sido ensinado ou decidido em sessões "
                        "anteriores, ou quando perguntar se você lembra. "

                        "Se recall não encontrar nada relevante, não invente. "

                        "Quando recall encontrar uma memória simples, "
                        "responda diretamente com o fato recuperado. "

                        "Quando remember salvar uma memória, confirme "
                        "de forma curta e natural. "

                        "Quando precisar de uma ferramenta, chame-a "
                        "imediatamente e silenciosamente. "

                        "NÃO fale antes da ferramenta. "

                        "Não diga frases de preenchimento como "
                        "'deixa eu pensar', 'vou verificar', "
                        "'vou dar uma olhada', 'vamos ver', "
                        "'um momento' ou equivalentes. "

                        "Não explique que vai usar uma ferramenta. "

                        "Primeiro execute a ferramenta. "

                        "Somente depois de receber o resultado da ferramenta "
                        "produza a resposta falada ao Luciano. "

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
                "🟢 AURI v0.5.5.1 ONLINE"
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
