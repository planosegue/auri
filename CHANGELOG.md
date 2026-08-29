# AURI Changelog

## v0.5.5 — Planned

Persistent Memory.

Planejado:

- SQLite
- remember()
- recall()
- deduplicação
- memória entre sessões

## v0.5.4.1 — Stable Baseline

Adicionado e validado:

- Real Tools
- automatic function calling
- tool chaining
- automatic barge-in
- clear_player
- async speaker finalization
- WebSocket event loop não bloqueante
- antenna rest offset
- volume persistente
- visão
- Web Search contextual

Validado:

- interrupção durante resposta
- search_web
- look
- múltiplas pesquisas sequenciais
- contexto entre tools

Known issue:

- jitter da antena direita

## v0.5.4

Primeira implementação Real Tools.

Problemas identificados:

- barge-in manual não confiável
- event_loop bloqueante
- WebSocket keepalive timeout

## v0.5.3

- Web Search
- Vision
- Volume
- router baseado em intenção textual

## v0.5.2

Primeira integração Vision.

## v0.5.1

Baseline inicial estável.

- conversa contínua
- VAD
- português
- expressões
- anti-echo

## v0.5

Continuous conversation.

## v0.4

Correção do pipeline de áudio.

16 → 24 kHz input.
24 → 16 kHz output.

## v0.3

Diagnóstico de STT.

## v0.2

Primeira resposta Realtime pelo speaker.

## v0.1

Conexão ao Reachy e Expression Engine.
