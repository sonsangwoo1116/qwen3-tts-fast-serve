"""
Async engine step loop: runs talker and predictor steps on the event loop thread
(when they have work), publishes token/done to ZMQ. No threads — step() runs on
the same thread as the asyncio event loop so CUDA graphs are not hurt.
"""

import asyncio
from typing import Any


async def run_engine_loop(
    talker_llm: Any,
    predictor_llm: Any,
    zmq_bridge: Any,
) -> None:
    """
    Run forever as an asyncio task. When talker or predictor have pending work,
    run step() on the current (event loop) thread and publish to ZMQ.
    Yields to the event loop when idle via asyncio.sleep(0.001).
    Run predictor first when both have work so the consumer can get predictor token.
    """
    while True:
        has_talker = bool(talker_llm.scheduler.waiting or talker_llm.scheduler.running)
        has_predictor = bool(predictor_llm.scheduler.waiting or predictor_llm.scheduler.running)

        if has_predictor:
            try:
                outputs, _ = predictor_llm.step()
                for request_id, seq_id, token_ids in outputs:
                    zmq_bridge.publish_token("predictor", request_id, token_ids, None)
            except Exception as e:
                raise e
            await asyncio.sleep(0)
            continue
        if has_talker:
            try:
                _, _, outputs_all = talker_llm.step_with_outputs()
                for tup in outputs_all:
                    request_id, seq_id, token_ids, hidden_states, is_finished = tup
                    zmq_bridge.publish_token("talker", request_id, token_ids, hidden_states)
                    if is_finished:
                        zmq_bridge.publish_done("talker", request_id)
            except Exception as e:
                raise e
            await asyncio.sleep(0)
            continue
        await asyncio.sleep(0.001)
