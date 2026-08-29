from pathlib import Path

SOURCE = Path("/home/pollen/auri/auri_voice_v5_4_1.py")
TARGET = Path("/home/pollen/auri/auri_voice_v5_5.py")

code = SOURCE.read_text(encoding="utf-8")


def replace_once(old, new, description):

    global code

    if old not in code:
        raise RuntimeError(
            f"Não encontrei o trecho necessário: {description}"
        )

    code = code.replace(
        old,
        new,
        1
    )


# ============================================================
# IMPORTS
# ============================================================

replace_once(
    "import re\nimport subprocess\nimport time\n",
    """import re
import sqlite3
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
""",
    "imports",
)


# ============================================================
# CONSTANTES
# ============================================================

replace_once(
    "BARGE_IN_ARM_DELAY = 0.50\n",
    """BARGE_IN_ARM_DELAY = 0.50

# ============================================================
# MEMORY
# ============================================================

MEMORY_DB = Path(
    "/home/pollen/auri/data/auri_memory.db"
)

MEMORY_MAX_RESULTS = 8
MEMORY_MAX_FACT_CHARS = 800
""",
    "constantes de memória",
)


# ============================================================
# STATE
# ============================================================

replace_once(
    """        self.vision_processing = False
        self.web_processing = False
""",
    """        self.vision_processing = False
        self.web_processing = False
        self.memory_processing = False
""",
    "estado memory_processing",
)


# ============================================================
# MEMORY ENGINE
# ============================================================

memory_engine = r'''

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
'''

marker = "# ============================================================\n# WEB SEARCH\n# ============================================================"

if marker not in code:
    raise RuntimeError(
        "Não encontrei o ponto de inserção antes de WEB SEARCH."
    )

code = code.replace(
    marker,
    memory_engine + "\n\n" + marker,
    1
)


# ============================================================
# TOOL EXECUTOR
# ============================================================

replace_once(
    """    if name == "set_volume":

        return execute_volume_tool(
            args
        )


    return {
""",
    """    if name == "set_volume":

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
""",
    "executor remember/recall",
)


# ============================================================
# MICROPHONE
# ============================================================

replace_once(
    """            state.vision_processing
            or state.web_processing
            or (
""",
    """            state.vision_processing
            or state.web_processing
            or state.memory_processing
            or (
""",
    "pausa de áudio durante memória",
)


# ============================================================
# MEMORY TOOLS
# ============================================================

memory_tools = r'''

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
'''

main_marker = "# ============================================================\n# MAIN\n# ============================================================"

if main_marker not in code:
    raise RuntimeError(
        "Não encontrei o início de MAIN."
    )

code = code.replace(
    main_marker,
    memory_tools + "\n\n" + main_marker,
    1
)


# ============================================================
# IDENTIDADE / INSTRUCTIONS
# ============================================================

replace_once(
    """                        "Use set_volume quando Luciano pedir alteração "
                        "do volume físico. "
""",
    """                        "Use set_volume quando Luciano pedir alteração "
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
""",
    "instruções de memória",
)


# ============================================================
# STARTUP
# ============================================================

replace_once(
    """    client = AsyncOpenAI(
        api_key=api_key
    )


    current_volume = get_volume()
""",
    """    initialize_memory()


    client = AsyncOpenAI(
        api_key=api_key
    )


    current_volume = get_volume()
    memories_count = memory_stats()
""",
    "initialize_memory",
)


replace_once(
    """        f"📡 Antenna rest: "
        f"{ANTENNA_REST_OFFSET} rad"
    )
""",
    """        f"📡 Antenna rest: "
        f"{ANTENNA_REST_OFFSET} rad"
    )

    print(
        f"🧠 Memórias persistentes: "
        f"{memories_count}"
    )
""",
    "contador de memórias",
)


# ============================================================
# VERSION
# ============================================================

code = code.replace(
    "🤖 AURI v0.5.4.1",
    "🤖 AURI v0.5.5",
)

code = code.replace(
    " Real Tools + Stable Barge-in",
    " Real Tools + Persistent Memory",
)

code = code.replace(
    "🟢 AURI v0.5.4.1 ONLINE",
    "🟢 AURI v0.5.5 ONLINE",
)


# ============================================================
# WRITE
# ============================================================

TARGET.write_text(
    code,
    encoding="utf-8"
)

print("")
print("==============================")
print("🧠 AURI v0.5.5 Builder")
print("==============================")
print("")
print("✓ Origem:", SOURCE)
print("✓ Criado:", TARGET)
print("")
print("Próximo:")
print(
    "python -m py_compile "
    "/home/pollen/auri/auri_voice_v5_5.py"
)
