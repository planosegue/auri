# AURI Setup

## SSH

ssh pollen@reachy-mini.local

## Projeto

cd ~/auri

## Virtualenv

source .venv/bin/activate

## Python

python --version

Ambiente AURI originalmente criado com Python 3.13.5.

## OpenAI

Arquivo:

/home/pollen/auri/.env

Variável:

OPENAI_API_KEY

Permissão recomendada:

chmod 600 ~/auri/.env

## Executar baseline

python -m py_compile ~/auri/auri_voice_v5_4_1.py

python ~/auri/auri_voice_v5_4_1.py

## Reachy diagnostics

reachyminios_check

Daemon:

systemctl status reachy-mini-daemon --no-pager

Audio:

arecord -l
aplay -l

Volume:

amixer -c 0 get PCM,0
amixer -c 0 get PCM,1

## Git

git status
git add .
git commit -m "descricao"
git push

## Ambientes

Não instalar dependências experimentais em:

/venvs/apps_venv
/venvs/mini_daemon

Usar:

/home/pollen/auri/.venv
