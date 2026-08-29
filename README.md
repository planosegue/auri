# AURI

**AURI — Adaptive Unified Robotic Intelligence**

AURI é uma inteligência artificial robótica desenvolvida inicialmente sobre o Reachy Mini Wireless, com arquitetura planejada para separar inteligência, percepção, memória e ferramentas do corpo físico.

O Reachy Mini é o primeiro corpo da AURI. A arquitetura futura deverá permitir outros bridges robóticos, incluindo Unitree.

## Estado atual

- Versão estável atual: v0.5.4.1
- Próxima versão planejada: v0.5.5 — Persistent Memory
- Hardware atual: Reachy Mini Wireless
- Projeto: /home/pollen/auri
- Ambiente Python: /home/pollen/auri/.venv

## Capacidades validadas

- OpenAI Realtime
- conversa contínua
- português brasileiro
- Server VAD
- contexto conversacional
- respostas pelo speaker do Reachy
- barge-in
- visão
- Web Search
- controle de volume
- Real Tools
- tool chaining

## Real Tools

Ferramentas atuais:

- search_web(query)
- look(question)
- set_volume(action, percent)

A partir da v0.5.4 o próprio modelo Realtime decide semanticamente quando utilizar uma ferramenta.

## Barge-in

Validado na v0.5.4.1.

Fluxo:

AURI falando
→ Luciano começa a falar
→ Server VAD detecta
→ resposta é cancelada
→ clear_player()
→ speaker para
→ AURI entra em listening

## Tool chaining

Validado na v0.5.4.1.

AURI consegue executar ferramentas sequencialmente quando necessário.

Caso validado:

Oura
→ search_web para identificação
→ search_web para preço
→ resposta final

## Arquitetura

AURI Core
→ OpenAI Realtime
→ Conversation Context
→ Tool Planning

Tools:
- Vision
- Web Search
- Volume
- futura Memory

Robot Bridge:
- Reachy Mini atualmente
- futuro Unitree Bridge

## Áudio

Pipeline validado:

Reachy microphone
→ 16 kHz stereo float32
→ canal 0
→ resample 16 para 24 kHz
→ PCM16
→ OpenAI Realtime
→ PCM16 24 kHz
→ resample 24 para 16 kHz
→ Reachy speaker

A correção de sample rate foi fundamental para reconhecimento correto de português.

## Playback

response.done NÃO significa que o speaker terminou fisicamente.

Na v0.5.4.1 a espera do speaker ocorre em uma task assíncrona separada para não bloquear o WebSocket.

## Volume

PCM,0 = volume global controlado pela AURI.

PCM,1 deve permanecer em 100%.

AURI lê o volume físico atual no startup.

## Antenas

Existe jitter persistente na antena direita depois que AURI assume controle.

ANTENNA_REST_OFFSET = 0.17 não resolveu completamente.

Status: OPEN ISSUE.

## Memória

Em desenvolvimento para v0.5.5.

SQLite validado em:

/home/pollen/auri/data/auri_memory.db

Tools planejadas:

- remember()
- recall()
- forget() futuramente

O banco de memória não deve ser enviado ao Git.

## Segurança

Nunca versionar:

- .env
- .venv/
- data/
- *.db
- tokens
- API keys

## Executar

ssh pollen@reachy-mini.local

cd ~/auri

source .venv/bin/activate

python -m py_compile ~/auri/auri_voice_v5_4_1.py

python ~/auri/auri_voice_v5_4_1.py

## GitHub

Repositório:

planosegue/auri

Fluxo normal:

git status
git add .
git commit -m "descricao"
git push

Baseline atual:

v0.5.4.1

## Physical Design

Conceito escolhido:

AURI Futuristic

Estratégia Physical v1:

Overlay Kit totalmente reversível.

Objetivo:

- head overlay
- antenna sleeves
- body ring
- AURI branding

Regra:

zero peças originais do Reachy removidas na Physical v1.
