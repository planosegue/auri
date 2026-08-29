import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI


MODEL = "gpt-realtime-2.1-mini"


async def main():

    load_dotenv(
        "/home/pollen/auri/.env"
    )

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("")
    print("==============================")
    print("🤖 AURI Real Tools Test")
    print("==============================")
    print("")

    async with client.realtime.connect(
        model=MODEL
    ) as connection:

        print(
            "✓ Realtime conectado"
        )

        await connection.session.update(

            session={

                "type": "realtime",

                "model": MODEL,

                "output_modalities": [
                    "text"
                ],

                "instructions": (
                    "Você é AURI. "
                    "Use as ferramentas disponíveis "
                    "quando forem necessárias. "
                    "Se Luciano pedir para olhar algo, "
                    "use look. "
                    "Se pedir pesquisa atual ou internet, "
                    "use search_web. "
                    "Se pedir alteração de volume, "
                    "use set_volume."
                ),

                "tools": [

                    {
                        "type": "function",

                        "name": "look",

                        "description": (
                            "Olha através da câmera física "
                            "do corpo robótico para responder "
                            "uma pergunta visual."
                        ),

                        "parameters": {

                            "type": "object",

                            "properties": {

                                "question": {
                                    "type": "string"
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

                        "name": "search_web",

                        "description": (
                            "Pesquisa informações atuais "
                            "na internet."
                        ),

                        "parameters": {

                            "type": "object",

                            "properties": {

                                "query": {
                                    "type": "string"
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

                        "name": "set_volume",

                        "description": (
                            "Altera o volume físico do "
                            "alto-falante do robô."
                        ),

                        "parameters": {

                            "type": "object",

                            "properties": {

                                "percent": {
                                    "type": "integer",
                                    "minimum": 10,
                                    "maximum": 100
                                }
                            },

                            "required": [
                                "percent"
                            ],

                            "additionalProperties": False,
                        },
                    },
                ],

                "tool_choice": "auto",
            }
        )

        print(
            "✓ Tools registradas"
        )

        await connection.conversation.item.create(

            item={

                "type": "message",

                "role": "user",

                "content": [

                    {
                        "type": "input_text",

                        "text": (
                            "Pesquise na internet quais são "
                            "as novidades mais recentes sobre "
                            "o Unitree R1."
                        ),
                    }
                ],
            }
        )

        await connection.response.create()

        print("")
        print(
            "→ Pergunta enviada"
        )
        print("")

        async for event in connection:

            print(
                "EVENT:",
                event.type
            )

            if event.type == \
                "response.function_call_arguments.done":

                print("")
                print(
                    "🎯 TOOL:",
                    event.name
                )

                print(
                    "ARGUMENTOS:",
                    event.arguments
                )

                try:

                    args = json.loads(
                        event.arguments
                    )

                    print(
                        "JSON:",
                        args
                    )

                except Exception:
                    pass

                print("")
                print(
                    "✓ Function calling validado"
                )

                break


asyncio.run(
    main()
)
