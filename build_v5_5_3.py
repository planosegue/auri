from pathlib import Path

SOURCE = Path(
    "/home/pollen/auri/auri_voice_v5_5_2.py"
)

TARGET = Path(
    "/home/pollen/auri/auri_voice_v5_5_3.py"
)

code = SOURCE.read_text(
    encoding="utf-8"
)


# ============================================================
# VERSION
# ============================================================

code = code.replace(
    "🤖 AURI v0.5.5.2",
    "🤖 AURI v0.5.5.3"
)

code = code.replace(
    "🟢 AURI v0.5.5.2 ONLINE",
    "🟢 AURI v0.5.5.3 ONLINE"
)

code = code.replace(
    " Natural Conversation + Persistent Memory",
    " Grounded Web + Persistent Memory"
)


# ============================================================
# WEB SEARCH PROMPT
# ============================================================

start_marker = '''            input=(
                "Pesquise na internet a solicitação abaixo. "'''

end_marker = '''                f"Consulta: {query}"
            ),'''

start = code.find(
    start_marker
)

end = code.find(
    end_marker,
    start
)


if start == -1:

    raise RuntimeError(
        "Não encontrei início do prompt Web."
    )


if end == -1:

    raise RuntimeError(
        "Não encontrei final do prompt Web."
    )


end += len(
    end_marker
)


new_web_prompt = '''            input=(
                "Você é o módulo de pesquisa factual em tempo real "
                "da AURI. "

                "Pesquise efetivamente na internet antes de responder. "

                "A consulta vem de Luciano, usuário brasileiro. "

                "PRIORIDADE DE FONTES: "

                "1. fabricante ou fonte oficial; "
                "2. documentação oficial; "
                "3. distribuidor ou varejista autorizado; "
                "4. veículos especializados confiáveis; "
                "5. outras fontes somente quando necessário. "

                "Não baseie uma conclusão importante em apenas uma "
                "fonte quando houver possibilidade razoável de conferir. "

                "Para PREÇOS: "

                "identifique primeiro o país e a moeda relevantes. "

                "Se a consulta não indicar outro país, considere "
                "Brasil como contexto principal. "

                "Procure primeiro preço oficial brasileiro em BRL/R$. "

                "Se existir preço oficial no Brasil, use-o como "
                "referência principal. "

                "Quando houver vários modelos ou configurações, "
                "obtenha exemplos concretos de pelo menos dois ou "
                "três modelos quando possível. "

                "Não invente uma faixa genérica antes de verificar "
                "preços concretos. "

                "Construa qualquer faixa de preço somente depois "
                "de observar exemplos reais. "

                "Nunca apresente preço em dólares como se fosse "
                "preço brasileiro. "

                "Se só houver preço internacional, diga explicitamente "
                "que é uma referência internacional e informe a moeda. "

                "Não faça conversão cambial aproximada sem necessidade. "

                "Diferencie preço oficial, preço de varejo e preço "
                "de mercado secundário/usado. "

                "Para PRODUTOS: "

                "confirme o nome e modelo oficial antes de responder. "

                "Se o usuário pronunciar ou transcrever o nome "
                "incorretamente, procure identificar o produto provável "
                "sem fingir certeza. "

                "Para COMPARAÇÕES: "

                "compare versões equivalentes e use especificações "
                "atuais verificadas. "

                "Para NOTÍCIAS e LANÇAMENTOS: "

                "priorize informações recentes e deixe clara a data "
                "quando ela for relevante. "

                "Não invente fatos, preços, modelos, disponibilidade "
                "ou especificações. "

                "Se as fontes forem insuficientes ou divergirem, "
                "diga isso claramente. "

                "A resposta será usada como contexto interno pela AURI. "

                "Seja factual e suficientemente detalhado para que "
                "ela possa responder corretamente, mas evite texto "
                "desnecessário. "

                "Responda em português brasileiro. "

                f"Consulta: {query}"
            ),'''


code = (
    code[:start]
    + new_web_prompt
    + code[end:]
)


# ============================================================
# INSTRUCTIONS — USO DO RESULTADO WEB
# ============================================================

needle = '''                        "Não leia tabelas inteiras ou listas extensas "
                        "em voz alta se uma síntese responder à pergunta. "
'''

replacement = '''                        "Não leia tabelas inteiras ou listas extensas "
                        "em voz alta se uma síntese responder à pergunta. "

                        "Ao receber resultado de search_web, baseie sua "
                        "resposta nos dados concretos retornados pela pesquisa. "

                        "Não substitua preços concretos encontrados por uma "
                        "faixa genérica criada por você. "

                        "Para preço de produto, prefira mencionar dois ou "
                        "três exemplos concretos e depois resumir a faixa. "

                        "Diferencie claramente preço oficial, varejo, usado "
                        "e referência internacional quando aplicável. "

                        "Se a pesquisa disser que determinado preço ou "
                        "informação não foi encontrado com confiança, "
                        "não preencha a lacuna com conhecimento presumido. "
'''


if needle not in code:

    raise RuntimeError(
        "Não encontrei instrução de síntese Web."
    )


code = code.replace(
    needle,
    replacement,
    1
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
    'MEMORY_DB',
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
print("🌐 AURI v0.5.5.3 Builder")
print("==============================")
print("")
print("✓ Origem:", SOURCE)
print("✓ Criado:", TARGET)
print("✓ Voice preservada")
print("✓ Memory preservada")
print("✓ Barge-in preservado")
print("✓ Real Tools preservadas")
print("✓ Web Grounding reforçado")
print("✓ Brasil/BRL priorizado")
print("✓ Fontes oficiais priorizadas")
print("")
