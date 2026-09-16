#!/usr/bin/env python3
"""Round-robin TCP load balancer for two local UAC/Pi model tunnels."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import sys


@dataclass
class Backend:
    host: str
    port: int
    active: int = 0


class BackendPool:
    def __init__(self, backends: list[Backend], max_connections: int) -> None:
        self.backends = backends
        self.max_connections = max_connections
        self.next_index = 0
        self.condition = asyncio.Condition()

    async def acquire(self) -> Backend:
        async with self.condition:
            while True:
                for offset in range(len(self.backends)):
                    index = (self.next_index + offset) % len(self.backends)
                    backend = self.backends[index]
                    if backend.active < self.max_connections:
                        backend.active += 1
                        self.next_index = (index + 1) % len(self.backends)
                        return backend
                await self.condition.wait()

    async def release(self, backend: Backend) -> None:
        async with self.condition:
            backend.active -= 1
            self.condition.notify_all()


async def copy_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    await asyncio.gather(writer.wait_closed(), return_exceptions=True)


async def handle_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    pool: BackendPool,
) -> None:
    backend = await pool.acquire()
    peer = client_writer.get_extra_info("peername")
    print(
        f"connection {peer} -> {backend.host}:{backend.port} "
        f"(active={backend.active})",
        flush=True,
    )
    try:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                backend.host, backend.port
            )
        except OSError as exc:
            print(
                f"cannot connect to {backend.host}:{backend.port}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            await close_writer(client_writer)
            return

        try:
            transfers = {
                asyncio.create_task(copy_stream(client_reader, upstream_writer)),
                asyncio.create_task(copy_stream(upstream_reader, client_writer)),
            }
            _, pending = await asyncio.wait(
                transfers, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*transfers, return_exceptions=True)
        finally:
            await close_writer(upstream_writer)
            await close_writer(client_writer)
    finally:
        await pool.release(backend)


def parse_backend(value: str) -> Backend:
    try:
        host, raw_port = value.rsplit(":", 1)
        port = int(raw_port)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"backend must have HOST:PORT form, got {value!r}"
        ) from exc
    if not host or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"invalid backend: {value!r}")
    return Backend(host, port)


async def run(args: argparse.Namespace) -> None:
    pool = BackendPool(args.backend, args.max_connections)
    server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, pool),
        args.listen_host,
        args.listen_port,
    )
    targets = ", ".join(f"{item.host}:{item.port}" for item in args.backend)
    print(
        f"listening on {args.listen_host}:{args.listen_port}; "
        f"backends: {targets}; max {args.max_connections} connections each",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument(
        "--backend",
        action="append",
        type=parse_backend,
        required=True,
        help="Repeat for each HOST:PORT model tunnel.",
    )
    parser.add_argument("--max-connections", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.listen_port <= 65535:
        parser.error("--listen-port must be between 1 and 65535")
    if args.max_connections < 1:
        parser.error("--max-connections must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
