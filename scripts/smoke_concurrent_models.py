"""Real two-server overlap test, without tools or the speech/vision stack.

Close ASTA first. No external server is killed, no generated code is executed.
This checks model residency and overlap only; full speech-stack VRAM still needs
checking in ASTA itself. A successful HTTP overlap is not proof of parallel GPU kernels.
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from core import Kernel
    from core.config import load_local_environment
    from ai.llm_provider import create_llm_provider
    load_local_environment()
    if os.getenv('ASTA_MODEL_ROUTER','0') != '1' or os.getenv('ASTA_MODEL_MODE') != 'dual':
        raise RuntimeError('Configure ASTA_MODEL_ROUTER=1 and ASTA_MODEL_MODE=dual first.')
    kernel = Kernel()
    threads = []
    try:
        chat = create_llm_provider(kernel=kernel, purpose='chat')
        coder = create_llm_provider(kernel=kernel, purpose='agent')
        runtime = kernel.model_runtime
        if any(m._server_is_ready() for m in runtime.managers.values()):
            raise RuntimeError('Close ASTA and external text servers before this smoke test.')
        for profile, client in [('general',chat),('coding',coder)]:
            runtime.execute(profile, client._engine(profile), lambda e: e.warmup())
        if not all(m.owned and m.running for m in runtime.managers.values()):
            raise RuntimeError('Both ASTA-owned models must be resident.')
        # The event fires within the coding request, after receiving a sentence.
        # If it finishes too quickly to overlap, report that instead of claiming success.
        first_output = threading.Event()
        timings, replies, failures = {}, {}, []
        def coding():
            try:
                engine = coder._engine('coding')
                coder._configure(engine, 'coding')
                engine.reset_conversation()
                timings['coding_start'] = time.perf_counter()
                replies['coding'] = runtime.execute('coding', engine, lambda e: e.generate_response(
                    'Write a Python CLI todo application with JSON persistence, argument parsing, input validation, and several pytest tests. Return the complete code with a short explanation.',
                    on_sentence=lambda sentence: first_output.set()))
                timings['coding_end'] = time.perf_counter()
            except Exception as exc:
                failures.append(type(exc).__name__+': '+str(exc)); first_output.set()
        thread = threading.Thread(target=coding, name='smoke-coding')
        threads.append(thread); thread.start()
        if not first_output.wait(120):
            raise RuntimeError('Coding produced no sentence within 120 seconds.')
        if failures:
            raise RuntimeError(failures[0])
        if not thread.is_alive():
            raise RuntimeError('Coding finished before overlap could be tested; no concurrency result established.')
        timings['general_start'] = time.perf_counter()
        replies['general'] = chat.generate_response('What is the capital of India? Answer in one sentence.')
        timings['general_end'] = time.perf_counter()
        thread.join(timeout=180)
        if thread.is_alive():
            raise RuntimeError('Coding did not finish within the smoke limit.')
        if failures:
            raise RuntimeError(failures[0])
        if not replies.get('coding') or not replies.get('general') or replies['general'].startswith("I couldn't use"):
            raise RuntimeError('A model returned no usable response.')
        if timings['general_end'] >= timings['coding_end']:
            raise RuntimeError('Chat did not finish before coding; interactive overlap was not demonstrated.')
        if not all(m.running for m in runtime.managers.values()):
            raise RuntimeError('A resident model stopped during overlap.')
        print(json.dumps({'general_seconds':round(timings['general_end']-timings['general_start'],2),
            'coding_seconds':round(timings['coding_end']-timings['coding_start'],2),
            'chat_finished_while_coding_active':True,
            'both_servers_resident':True,'general_reply':replies['general']},indent=2),flush=True)
    finally:
        runtime = kernel.model_runtime
        if runtime is not None:
            for client in tuple(runtime.clients):
                client.cancel()
        for thread in threads:
            thread.join(timeout=5)
        kernel.shutdown()
        if runtime and any(m.owned for m in runtime.managers.values()):
            raise RuntimeError('An owned model process could not be stopped.')
    print('ASTA_CONCURRENT_MODELS_SMOKE_OK',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Concurrent smoke failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
