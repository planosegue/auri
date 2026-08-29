# AURI — AI CONTEXT

Leia este documento integralmente antes de modificar o projeto.

## Missão

AURI é uma inteligência artificial robótica.

Atualmente utiliza Reachy Mini Wireless como corpo, mas a inteligência deve permanecer desacoplada do hardware para permitir integração futura com outros robôs.

## Estado atual

Versão estável:

v0.5.4.1

Próximo milestone:

v0.5.5 Persistent Memory

## Ambiente

Host:

reachy-mini.local

SSH:

pollen@reachy-mini.local

Projeto:

/home/pollen/auri

Virtualenv:

/home/pollen/auri/.venv

Python AURI:

3.13.5

Reachy daemon:

1.9.0

## Ambientes oficiais Reachy

Existem:

/venvs/mini_daemon
/venvs/apps_venv

NÃO instalar dependências experimentais nesses ambientes.

Um upgrade do pacote OpenAI no apps_venv já causou conflito com o Conversation App oficial.

Todas as dependências experimentais do AURI devem permanecer em:

/home/pollen/auri/.venv

## Audio Pipeline

Reachy input:

float32 stereo 16 kHz

Usar canal 0:

samples[:, 0]

Converter:

16 kHz → 24 kHz
float32 → PCM16

Enviar ao Realtime.

Output:

PCM16 24 kHz
→ float32
→ 16 kHz
→ Reachy speaker

Essa correção resolveu reconhecimento incorreto de português.

## Realtime

Modelo atual:

gpt-realtime-2.1-mini

AURI usa Server VAD.

v0.5.4.1 utiliza:

create_response=True
interrupt_response=True

## Barge-in

Validado.

Quando Luciano fala durante uma resposta:

1. Server VAD detecta speech_started.
2. OpenAI cancela a resposta.
3. AURI executa clear_player().
4. áudio físico restante é descartado.
5. AURI entra em listening.
6. nova fala é processada.

Não voltar ao cancelamento manual usado na v0.5.4.

## WebSocket

Problema encontrado na v0.5.4:

keepalive ping timeout

Causa:

event_loop bloqueado esperando speaker ou tool.

Correção na v0.5.4.1:

operações demoradas rodam com asyncio.create_task().

O event_loop deve permanecer livre para consumir eventos continuamente.

NUNCA colocar sleeps longos ou chamadas demoradas de Web/Vision bloqueando o event_loop.

## Real Tools

Tools atuais:

search_web(query)
look(question)
set_volume(action, percent)

Realtime decide semanticamente quando utilizá-las.

Não voltar para roteadores baseados em listas de palavras.

## Tool chaining

Validado.

AURI consegue executar mais de uma tool para concluir uma tarefa.

Caso validado:

Oura
→ search_web identificação
→ search_web preço
→ resposta final

## Vision

Camera:

numpy.ndarray
shape=(720,1280,3)
dtype=uint8

Reachy entrega BGR.

Converter:

frame_rgb = frame[:, :, ::-1]

Vision deve produzir percepção curta.

Evitar respostas visuais excessivamente longas.

## Volume

PCM,0 = volume global.

PCM,1 deve permanecer 100%.

AURI deve ler o volume atual no startup.

Não redefinir DEFAULT_VOLUME arbitrariamente.

## Antenna Issue

Existe jitter persistente na antena direita.

Comportamento observado:

- robô reiniciado: não treme
- AURI inicia: começa jitter
- AURI encerra: jitter continua
- reboot elimina temporariamente

Tentativa:

ANTENNA_REST_OFFSET = 0.17

Não resolveu completamente.

Status:

OPEN ISSUE.

## Memory

SQLite foi validado.

Database:

/home/pollen/auri/data/auri_memory.db

Teste mostrou persistência correta.

Problema identificado:

INSERT simples gera duplicatas.

v0.5.5 deve implementar deduplicação.

Memória planejada:

remember
recall

Futuro:

forget

Categorias sugeridas:

- identity
- project
- preference
- decision
- technical
- relationship
- general

Nunca armazenar:

- passwords
- API keys
- tokens
- segredos

## Git

Repository:

planosegue/auri

Baseline:

v0.5.4.1

Antes de grandes alterações:

git status
git add .
git commit
git push

## Physical Design

Conceito escolhido:

AURI Futuristic.

Estratégia:

Overlay Kit reversível.

Physical v1 não deve exigir desmontagem do Reachy.

Objetivo:

- head overlay
- antenna sleeves
- body ring
- AURI branding

Regra:

zero peças originais removidas na v1.

## Filosofia

Preservar sempre:

- estabilidade
- expressividade
- antenas no listening
- barge-in
- contexto
- arquitetura modular
- capacidade de rollback

Nunca destruir uma baseline funcional para adicionar uma feature nova.

## Próximo passo

Implementar v0.5.5 Persistent Memory em cima da v0.5.4.1.
