FROM rust:1.83.0-slim-bookworm@sha256:540c902e99c384163b688bbd8b5b8520e94e7731b27f7bd0eaa56ae1960627ab AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo rustc --locked --release -- -C target-feature=+crt-static

FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc AS rust
COPY --from=build /build/target/release/mechbot /bot/mechbot
USER 65534:65534
ENTRYPOINT ["/bot/mechbot"]

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS b2c
RUN pip install --no-cache-dir numpy==2.4.3
WORKDIR /bot
COPY b2c/ ./
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
