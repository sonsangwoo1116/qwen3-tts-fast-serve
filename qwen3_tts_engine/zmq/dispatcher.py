"""
ZMQ SUB dispatcher: subscribes to talker/ and predictor/, parses request_id from topic,
dispatches (engine_type, msg_type, payload_dict) to per-request_id asyncio queues.

Uses a dedicated thread for blocking ZMQ recv (so it is already waiting before the
first publish, avoiding slow-joiner). An asyncio task reads from the thread's inbox
and dispatches to request_queues.
"""

import asyncio
import queue
import threading
from typing import Any

from qwen3_tts_engine.zmq.output_bridge import deserialize_token_payload

try:
    import zmq
except ImportError:
    zmq = None


def _ensure_zmq():
    if zmq is None:
        raise ImportError("pyzmq is required. Install with: pip install pyzmq")


def _recv_thread(connect_address: str, inbox: queue.Queue) -> None:
    """Run in a dedicated thread: blocking recv, put (request_id, engine_type, msg_type, payload_dict) in inbox."""
    _ensure_zmq()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(connect_address)
    sub.setsockopt(zmq.SUBSCRIBE, b"talker/")
    sub.setsockopt(zmq.SUBSCRIBE, b"predictor/")
    try:
        while True:
            msg = sub.recv_multipart()
            if len(msg) < 3:
                continue
            topic_b, msg_type_b, payload_b = msg[0], msg[1], msg[2]
            topic = topic_b.decode("utf-8")
            msg_type = msg_type_b.decode("utf-8")
            parts = topic.split("/", 1)
            engine_type = parts[0] if parts else ""
            request_id = parts[1] if len(parts) > 1 else ""
            if msg_type == "token":
                payload_dict = deserialize_token_payload(payload_b)
            else:
                payload_dict = {}
            inbox.put((request_id, engine_type, msg_type, payload_dict))
    finally:
        sub.close()
        ctx.term()


async def run_dispatch_loop(
    inbox: queue.Queue,
    request_queues: dict[str, Any],
    queues_lock: asyncio.Lock,
) -> None:
    """Asyncio task: get from inbox (via executor) and put into request_queues[request_id]."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(None, inbox.get)
        except Exception:
            continue
        if item is None:
            break
        request_id, engine_type, msg_type, payload_dict = item
        async with queues_lock:
            q = request_queues.get(request_id)
        if q is not None:
            try:
                q.put_nowait((engine_type, msg_type, payload_dict))
            except Exception:
                pass


def start_dispatcher_thread(connect_address: str) -> tuple[threading.Thread, queue.Queue]:
    """Start the ZMQ recv thread and return (thread, inbox). Caller must run run_dispatch_loop(inbox, ...) as asyncio task."""
    _ensure_zmq()
    inbox = queue.Queue()
    thread = threading.Thread(target=_recv_thread, args=(connect_address, inbox), daemon=True)
    thread.start()
    return thread, inbox
