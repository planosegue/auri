# AURI Troubleshooting

## Português reconhecido como inglês ou francês

Causa identificada:

sample rate incorreto.

Correção:

Reachy 16 kHz
→ resample 24 kHz
→ OpenAI.

Usar canal 0 do stereo.

## AURI fala sozinha depois da resposta

Causa:

speaker ainda reproduzindo áudio enquanto microfone volta ao Realtime.

Correções:

- assistant_speaking
- speaker timing
- anti-echo
- clear_player
- barge-in

## WebSocket timeout

Erro:

keepalive ping timeout

Causa:

event_loop bloqueado.

Correção:

tarefas demoradas devem usar asyncio.create_task().

Nunca bloquear event_loop esperando speaker, Vision ou Web Search.

## Barge-in

Configuração atual:

create_response=True
interrupt_response=True

Ao receber speech_started:

clear_player()

Não usar response.cancel manualmente na arquitetura atual.

## Volume baixo

Verificar:

amixer -c 0 get PCM,0
amixer -c 0 get PCM,1

PCM,1:

100%

PCM,0:

volume global.

## Antena direita tremendo

Status:

OPEN ISSUE.

Observado:

- sem AURI após reboot: estável
- iniciar AURI: jitter
- encerrar AURI: jitter continua
- reboot remove jitter

Offset 0.17 rad não resolveu completamente.

Necessário diagnóstico específico de servo, calibração ou holding.

## Vision

Camera:

1280x720 BGR.

Converter para RGB antes de Pillow.

## Memory duplicates

Teste inicial SQLite gerou duplicatas porque remember() usava INSERT simples.

v0.5.5 deve implementar deduplicação.
