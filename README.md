# Mechtoyale V2

Bot de torneo en Rust para `mechfight/1`. Detecta señales de Practice y FFA en cada partida. Busca duelos aislados en Practice y prioriza cubos, rotaciones y remates oportunistas en FFA.

El protocolo y las reglas oficiales están en [protocol.md](protocol.md). El mapa, el checker y los ejemplos de Python y TypeScript provienen del [kit oficial](mechfight-kit.md).

## Compilar y verificar

```sh
cargo test --locked --release
docker build --platform linux/amd64 -t mechtoyale:v2 .

python3 check/check.py --map-dir /arena -- \
  docker run --rm -i --network none --cpus 1 \
    --memory 512m --memory-swap 512m --pids-limit 128 --cap-drop ALL \
    --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 \
    -v "$PWD/arena:/arena:ro" mechtoyale:v2

python3 scripts/bench_latency.py
```

El benchmark usa 15 mechs, 40 proyectiles y 30 sonidos por observación. También comprueba la respuesta al tick más reciente de una ráfaga de 10 observaciones y el cese de fuego durante la regeneración. Activa `--debug` para validar el modo `DEBUG_DRAW=1`.

El checker reproduce observaciones grabadas. Las puntuaciones y la clasificación se miden al enviar un repositorio público a la pista de Mech Royale.