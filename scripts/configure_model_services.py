"""Update only model-service settings in the ignored local .env; never print secrets."""
from __future__ import annotations
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def update_env(path, values):
    lines = path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
    emitted = set()
    result = []
    for line in lines:
        match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=', line)
        key = match.group(1) if match else None
        if key in values:
            if key not in emitted:
                result.append(f'{key}={values[key]}')
                emitted.add(key)
        else:
            result.append(line)
    result.extend(f'{key}={value}' for key, value in values.items() if key not in emitted)
    path.write_text('\n'.join(result)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--general', required=True, type=Path)
    parser.add_argument('--coding', required=True, type=Path)
    parser.add_argument('--mode', choices=['dual','on_demand'], default='dual')
    args = parser.parse_args()
    for path in (args.general, args.coding):
        if not path.is_file() or path.suffix.lower() != '.gguf':
            parser.error('Both model paths must point to existing GGUF files.')
    if args.general.resolve() == args.coding.resolve():
        parser.error('Conversation and coding profiles require different files.')
    values = {
        'ASTA_MODEL_ROUTER':'1','ASTA_MODEL_MODE':args.mode,'ASTA_BACKGROUND_TASKS':'1',
        'ASTA_LLM_PROVIDER':'llama_cpp','ASTA_AGENT_MODE':'1',
        'ASTA_LLM_MODEL_PATH':args.general.resolve().as_posix(),
        'ASTA_LLM_MODEL':'', 'ASTA_LLM_BASE_URL':'http://127.0.0.1:8080/v1',
        'ASTA_LLM_CONTEXT_SIZE':'4096','ASTA_LLM_BATCH_SIZE':'256','ASTA_LLM_UBATCH_SIZE':'128',
        'ASTA_LLM_PARALLEL':'1','ASTA_LLM_TEMPERATURE':'0.1','ASTA_LLM_TOP_K':'50',
        'ASTA_LLM_TOP_P':'0.95','ASTA_LLM_MIN_P':'0','ASTA_LLM_REASONING':'off',
        'ASTA_LLM_ENABLE_THINKING':'0','ASTA_CODING_LLM_MODEL_PATH':args.coding.resolve().as_posix(),
        'ASTA_CODING_LLM_MODEL':'','ASTA_CODING_LLM_BASE_URL':'http://127.0.0.1:8081/v1',
        'ASTA_CODING_LLM_CONTEXT_SIZE':'4096','ASTA_CODING_LLM_BATCH_SIZE':'256',
        'ASTA_CODING_LLM_UBATCH_SIZE':'128','ASTA_CODING_LLM_PARALLEL':'1',
        'ASTA_CODING_LLM_TEMPERATURE':'0.6','ASTA_CODING_LLM_TOP_K':'20',
        'ASTA_CODING_LLM_TOP_P':'0.95','ASTA_CODING_LLM_MIN_P':'0',
        'ASTA_CODING_LLM_REASONING':'off','ASTA_CODING_LLM_ENABLE_THINKING':'0',
        'ASTA_CODING_LLM_MAX_OUTPUT_TOKENS':'2048','ASTA_CODING_AGENT_MAX_OUTPUT_TOKENS':'2048',
    }
    update_env(ROOT/'.env', values)
    print('Updated local .env model settings. Existing unrelated settings were preserved.')


if __name__ == '__main__':
    main()
