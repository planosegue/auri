import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI


MODEL = "gpt-realtime-2.1-mini"


async def main():
    load_dotenv("/home/pollen/auri/.env")

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("🤖 AURI Realtime")
    print("→ Conectando...")

    async with client.realtime.connect(model=MODEL) as connection:

        print("✓ WebSocket conectado")

        await connection.session.update(
            session={
                "type": "realtime",
                "model": MODEL,
                "output_modalities": ["text"],
                "instructions": (
                    "Você é AURI, uma inteligência artificial robótica. "
                    "Responda em português do Brasil. "
                    "Seja natural, inteligente e objetiva."
                ),
            }
        )

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Responda somente: "
                            "AURI Realtime conectado."
                        ),
                    }
                ],
            }
        )

        await connection.response.create()

        async for event in connection:

            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            elif event.type == "response.output_text.done":
                print()

            elif event.type == "response.done":
                print("✓ Sessão Realtime validada")
                break


asyncio.run(main())
