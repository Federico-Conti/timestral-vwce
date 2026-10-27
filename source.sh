#!/bin/sh

if [ ! -f .venv/bin/activate ]; then
    echo 'Ambiente .venv assente: crearne uno prima di eseguire source source.sh.' >&2
    return 1 2>/dev/null || exit 1
fi
. .venv/bin/activate
