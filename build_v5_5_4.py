from pathlib import Path

SOURCE = Path(
    "/home/pollen/auri/auri_voice_v5_5_3.py"
)

TARGET = Path(
    "/home/pollen/auri/auri_voice_v5_5_4.py"
)

code = SOURCE.read_text(
    encoding="utf-8"
)


def replace_once(old, new, name):

    global code

    if old not in code:
        raise RuntimeError(
            f"Não encontrei: {name}"
        )

    code = code.replace(
        old,
        new,
        1
    )


# ============================================================
# VERSION
# ============================================================

code = code.replace(
    "🤖 AURI v0.5.5.3",
    "🤖 AURI v0.5.5.4"
)

code = code.replace(
    "🟢 AURI v0.5.5.3 ONLINE",
    "🟢 AURI v0.5.5.4 ONLINE"
)

code = code.replace(
    " Grounded Web + Persistent Memory",
    " Transcript-Grounded Tools + Memory"
)


# ============================================================
# STATE
# ============================================================

replace_once(
    """        self.user_speaking = False
""",
    """        self.user_speaking = False

        # Transcrição final do turno atual.
        self.latest_user_transcript = ""

        # Quantas tools já foram executadas
        # desde a última fala do usuário.
        self.tool_chain_depth = 0
""",
    "estado transcript/tool_chain"
)


# ============================================================
# NOVO TURNO
# ============================================================

replace_once(
    """            state.user_speaking = True


            if state.assistant_speaking:
""",
    """            state.user_speaking = True

            # Novo turno do usuário.
            # Zeramos somente a profundidade da cadeia.
            state.tool_chain_depth = 0


            if state.assistant_speaking:
""",
    "reset tool_chain"
)


# ============================================================
# TRANSCRIÇÃO FINAL
# ============================================================

old = '''            if transcript:

                print(
                    "\\nVOCÊ:",
                    transcript
                )
'''

new = '''            if transcript:

                state.latest_user_transcript = (
                    transcript
                )

                print(
                    "\\nVOCÊ:",
                    transcript
                )

                print(
                    "📝 Source of truth:",
                    state.latest_user_transcript
                )
'''

replace_once(
    old,
    new,
    "captura transcrição final"
)


# ============================================================
# SEARCH_WEB — PRIMEIRA TOOL ANCORADA
# ============================================================

old = '''    if name == "search_web":

        return await search_web(
            client,
            args.get(
                "query",
                ""
            )
        )
'''

new = '''    if name == "search_web":

        model_query = args.get(
            "query",
            ""
        )

        # Na PRIMEIRA search_web do turno,
        # a transcrição final do usuário é
        # a fonte de verdade.
        #
        # Em pesquisas seguintes, preservamos
        # a query criada pelo modelo para
        # permitir tool chaining.
        if (
            state.tool_chain_depth == 0
            and state.latest_user_transcript
        ):

            query = (
                state.latest_user_transcript
            )

            print("")
            print(
                "📝 Web query ancorada "
                "na transcrição:"
            )
            print(
                "   ",
                query
            )

        else:

            query = model_query

            print("")
            print(
                "🔗 Web query de tool chaining:"
            )
            print(
                "   ",
                query
            )


        state.tool_chain_depth += 1


        return await search_web(
            client,
            query
        )
'''

replace_once(
    old,
    new,
    "search_web transcript-grounded"
)


# ============================================================
# OUTRAS TOOLS TAMBÉM CONTAM NA CADEIA
# ============================================================

old = '''    if name == "look":

        return await look(
'''

new = '''    if name == "look":

        state.tool_chain_depth += 1

        return await look(
'''

replace_once(
    old,
    new,
    "depth look"
)


old = '''    if name == "set_volume":

        return execute_volume_tool(
'''

new = '''    if name == "set_volume":

        state.tool_chain_depth += 1

        return execute_volume_tool(
'''

replace_once(
    old,
    new,
    "depth volume"
)


old = '''    if name == "remember":

        return remember_memory(
'''

new = '''    if name == "remember":

        state.tool_chain_depth += 1

        return remember_memory(
'''

replace_once(
    old,
    new,
    "depth remember"
)


old = '''    if name == "recall":

        return recall_memory(
'''

new = '''    if name == "recall":

        state.tool_chain_depth += 1

        return recall_memory(
'''

replace_once(
    old,
    new,
    "depth recall"
)


# ============================================================
# GARANTIAS
# ============================================================

checks = [
    '"voice": "marin"',
    '"name": "search_web"',
    '"name": "look"',
    '"name": "remember"',
    '"name": "recall"',
    '"interrupt_response": True',
    "latest_user_transcript",
    "tool_chain_depth",
]


for check in checks:

    if check not in code:

        raise RuntimeError(
            "Garantia ausente: "
            + check
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
print("📝 AURI v0.5.5.4 Builder")
print("==============================")
print("")
print("✓ Origem:", SOURCE)
print("✓ Criado:", TARGET)
print("✓ Transcript grounding")
print("✓ Tool chaining preservado")
print("✓ Memory preservada")
print("✓ Barge-in preservado")
print("✓ Voice marin preservada")
print("")
