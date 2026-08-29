import sqlite3
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(
    "/home/pollen/auri/data/auri_memory.db"
)


def connect():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


def initialize():

    with connect() as conn:

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

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memories_subject
            ON memories(subject)
            """
        )

        conn.commit()


def remember(
    subject,
    fact,
    category="general",
    importance=5
):

    now = datetime.now(
        timezone.utc
    ).isoformat()

    with connect() as conn:

        cursor = conn.execute(
            """
            INSERT INTO memories (
                subject,
                fact,
                category,
                importance,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                subject,
                fact,
                category,
                importance,
                now,
                now,
            )
        )

        conn.commit()

        memory_id = cursor.lastrowid

    return {
        "ok": True,
        "memory_id": memory_id,
        "fact": fact,
    }


def recall(query, limit=10):

    query = query.strip()

    pattern = (
        "%"
        + query
        + "%"
    )

    with connect() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                subject,
                fact,
                category,
                importance,
                created_at
            FROM memories
            WHERE
                subject LIKE ?
                OR fact LIKE ?
                OR category LIKE ?
            ORDER BY
                importance DESC,
                id DESC
            LIMIT ?
            """,
            (
                pattern,
                pattern,
                pattern,
                limit,
            )
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def show_all():

    with connect() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                subject,
                fact,
                category,
                importance
            FROM memories
            ORDER BY id
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# TEST
# ============================================================

initialize()

print("")
print("==============================")
print("🧠 AURI Memory Test")
print("==============================")
print("")

result = remember(
    subject="Luciano",
    fact=(
        "Luciano está desenvolvendo "
        "o projeto AURI usando um "
        "Reachy Mini Wireless."
    ),
    category="project",
    importance=9,
)

print(
    "✓ Memória salva:",
    result
)

result = remember(
    subject="AURI",
    fact=(
        "AURI utiliza atualmente "
        "um Reachy Mini Wireless "
        "como corpo robótico."
    ),
    category="identity",
    importance=10,
)

print(
    "✓ Memória salva:",
    result
)

print("")
print(
    "🔎 Buscando: Reachy"
)

for memory in recall(
    "Reachy"
):

    print(
        "-",
        memory["fact"]
    )

print("")
print(
    "📚 Todas as memórias:"
)

for memory in show_all():

    print(
        memory["id"],
        memory["subject"],
        "→",
        memory["fact"]
    )

print("")
print(
    "✓ Memory Engine funcionando"
)
