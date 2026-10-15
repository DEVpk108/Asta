"""Final-batch regressions: no live models, accounts, microphone or camera."""
import ast
import csv
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from ai.wakeword.audio_contract import normalize_wav, validate_wav, WAKEWORD_SAMPLE_RATE
from core.applications.manager import ApplicationManager, ApplicationRecord, _dedupe
from core.contracts import ToolRequest
from core.quick_answers import quick_math, _evaluate_arithmetic
from core.tools.builtin import StopProcessTool

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('text,expected', [
    ('what is ten plus negative five', "That's 5."),
    ('what is 10 * -5', "That's -50."),
    ('what is negative five plus two', "That's -3."),
    ('what is ten minus negative five', "That's 15."),
    ('what is minus five times minus two', "That's 10."),
    ('what is two times negative three plus four', "That's -2."),
    ('what is one hundred and five plus two', "That's 107."),
    ('what is negative five squared', "That's 25."),
    ('what is cube root of negative eight', "That's -2."),
    ('what is square root of negative nine', None),
    ('what is two to the power of three', "That's 8."),
    ('what is ten divided by zero', "That can't be calculated, because it divides by zero."),
])
def test_spoken_arithmetic_and_negatives(text, expected):
    assert quick_math(text) == expected


@pytest.mark.parametrize('expression', [
    '__import__("os").system("echo bad")', 'x+1', '(lambda:1)()',
    '1 .__class__', '[1][0]', 'True+1', '2**1000000',
    '2**(2**64)', '1e100', 'float("inf")', '2**.5+1j',
])
def test_arithmetic_ast_rejects_non_arithmetic_or_unbounded_input(expression):
    with pytest.raises((ValueError, SyntaxError, OverflowError, TypeError)):
        _evaluate_arithmetic(expression)


def test_quick_math_has_no_eval_call():
    tree = ast.parse((ROOT/'core/quick_answers.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'eval' for n in ast.walk(tree))


def wav_fixture(path, rate=16000, channels=1, width=2, frames=1600):
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b'\0'*(frames*channels*width))
    return path


@pytest.mark.parametrize('rate,channels,width,frames', [
    (16000, 1, 2, 1600), (22050, 1, 2, 2205), (16000, 2, 2, 1600),
    (16000, 1, 1, 1600), (16000, 1, 2, 0),
])
def test_shared_wav_contract(tmp_path, rate, channels, width, frames):
    path = wav_fixture(tmp_path/'clip.wav', rate, channels, width, frames)
    if (rate, channels, width) == (16000, 1, 2) and frames:
        assert validate_wav(path) == path
    else:
        with pytest.raises(ValueError):
            validate_wav(path)


def test_native_rate_normalization_is_noop(tmp_path, monkeypatch):
    path = wav_fixture(tmp_path/'clip.wav')
    before = path.read_bytes()
    monkeypatch.setattr('ai.wakeword.audio_contract.subprocess.run', lambda *a, **kw: pytest.fail('unexpected conversion'))
    assert normalize_wav(path) == path
    assert path.read_bytes() == before


def test_failed_normalization_preserves_input_and_cleans_temporary(tmp_path, monkeypatch):
    path = wav_fixture(tmp_path/'clip.wav', rate=22050)
    before = path.read_bytes()
    monkeypatch.setattr('ai.wakeword.audio_contract.shutil.which', lambda name: 'ffmpeg')
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, 'ffmpeg')
    monkeypatch.setattr('ai.wakeword.audio_contract.subprocess.run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        normalize_wav(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.asta-wakeword-*'))


def test_missing_ffmpeg_is_actionable_and_preserves_input(tmp_path, monkeypatch):
    path = wav_fixture(tmp_path/'clip.wav', rate=22050)
    before = path.read_bytes()
    monkeypatch.setattr('ai.wakeword.audio_contract.shutil.which', lambda name: None)
    with pytest.raises(RuntimeError, match='FFmpeg'):
        normalize_wav(path)
    assert path.read_bytes() == before


def test_real_ffmpeg_resampling_keeps_duration_not_just_header(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('offline training FFmpeg not installed')
    path = wav_fixture(tmp_path/'clip.wav', rate=22050, frames=22050)
    normalize_wav(path)
    validate_wav(path)
    with wave.open(str(path), 'rb') as audio:
        assert audio.getnframes() == 16000
        assert audio.getnframes()/audio.getframerate() == 1.0


def test_training_constants_use_one_contract_and_installed_dependency():
    found = 0
    for path in (ROOT/'ai/wakeword/training').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                    'EXPECTED_SAMPLE_RATE', 'TARGET_SAMPLE_RATE', 'SAMPLE_RATE', 'TARGET_SR'} for t in node.targets):
                assert isinstance(node.value, ast.Name) and node.value.id == 'WAKEWORD_SAMPLE_RATE', path
                found += 1
    assert found >= 8
    feature_source = (ROOT/'ai/wakeword/training/10_extract_positive_features.py').read_text(encoding='utf-8')
    assert 'sys.path.insert' not in feature_source
    assert WAKEWORD_SAMPLE_RATE == 16000
    assert 'phrases=None' in (ROOT/'ai/wakeword/training/01_generate_dataset.py').read_text(encoding='utf-8')


def test_manifest_membership_loads_once_and_updates_after_append(tmp_path, monkeypatch):
    path = ROOT/'ai/wakeword/tools/piper_generator.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'_manifest_contains', '_write_manifest_entry'}]
    ns = dict(csv=csv, SPEAKER_WEIGHTS={0: 1})
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    manifest = tmp_path/'manifest.csv'
    manifest.write_text('filename,canonical_phrase,text,speaker_id,speaker_weight\nold.wav,hey,hey,0,1\n')
    obj = type('Manifest', (), {name: ns[name] for name in ('_manifest_contains', '_write_manifest_entry')})()
    obj.manifest_path = manifest
    obj._manifest_filenames = None
    opened = []
    original = Path.open
    def tracked(self, *args, **kwargs):
        if self == manifest:
            opened.append(args[0])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', tracked)
    for _ in range(100):
        assert obj._manifest_contains('old.wav')
        assert not obj._manifest_contains('new.wav')
    obj._write_manifest_entry('new.wav', 'hey', 'hey', 0)
    assert obj._manifest_contains('new.wav')
    assert opened.count('r') == 1 and opened.count('a') == 1


def test_negative_preparer_uses_actual_repository_root():
    path = ROOT/'ai/wakeword/training/05_prepare_negative_dataset.py'
    assert path.parents[3] == ROOT
    assert 'PROJECT_ROOT = Path(__file__).resolve().parents[3]' in path.read_text(encoding='utf-8')


def test_process_cache_expires_and_can_be_invalidated(monkeypatch):
    import core.applications.manager as module
    clock, calls = [10.0], []
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    manager = ApplicationManager(powershell_runner=lambda text: calls.append(text) or '[]', process_cache_seconds=.5)
    assert manager._discover_windows_processes() == []
    assert manager._discover_windows_processes() == []
    assert len(calls) == 1
    clock[0] += .6
    manager._discover_windows_processes()
    assert len(calls) == 2
    manager.invalidate_process_cache()
    manager._discover_windows_processes()
    assert len(calls) == 3


def test_process_cache_serializes_parallel_refresh():
    calls = []
    manager = ApplicationManager(powershell_runner=lambda text: calls.append(text) or '[]')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: manager._discover_windows_processes(), range(32)))
    assert len(calls) == 1


def test_dedupe_preserves_start_apps_priority_and_sorted_output():
    preferred = ApplicationRecord('Alpha', 'start', 'windows.start_apps')
    ordinary = ApplicationRecord('Alpha', 'path', 'shortcut')
    beta = ApplicationRecord('Beta', 'beta', 'shortcut')
    assert _dedupe([ordinary, beta, preferred, ordinary, preferred]) == (preferred, beta)


@pytest.mark.parametrize('windows', [False, True])
def test_stop_process_uses_platform_safe_argument_vector(monkeypatch, windows):
    calls = []
    monkeypatch.setattr('core.tools.builtin.platform.system', lambda: 'Windows' if windows else 'Linux')
    monkeypatch.setattr('core.tools.builtin.os.kill', lambda pid, sig: calls.append(('kill', pid)))
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr('core.tools.builtin.subprocess.run', run)
    result = StopProcessTool().execute(ToolRequest('system.stop_process', {'target': '1234'}, 'stop'))
    assert result.success
    if windows:
        assert calls[0][0] == ['taskkill', '/PID', '1234', '/T']
        assert calls[0][1]['shell'] is False and calls[0][1]['timeout'] == 15
    else:
        assert calls == [('kill', 1234)]


def test_stop_process_reports_taskkill_failure(monkeypatch):
    monkeypatch.setattr('core.tools.builtin.platform.system', lambda: 'Windows')
    monkeypatch.setattr('core.tools.builtin.subprocess.run', lambda *a, **kw: SimpleNamespace(
        returncode=1, stderr='access denied', stdout=''))
    result = StopProcessTool().execute(ToolRequest('system.stop_process', {'target': '1234'}, 'stop'))
    assert not result.success and 'access denied' in result.error


def test_workspace_does_not_rewrite_unchanged_persisted_fields(tmp_path):
    from core import Kernel
    from core.workspace_runtime import WorkspaceRuntimeModule
    kernel = Kernel()
    module = WorkspaceRuntimeModule(kernel, state_path=tmp_path/'workspace.json')
    saves = []
    module.state_store.save = lambda state: saves.append(dict(state))
    state = kernel.workspace_manager.snapshot()
    module.on_workspace_updated(workspace=state)
    state['environment'] = {'os': 'Windows'}
    module.on_workspace_updated(workspace=state)
    state['hardware'] = {'gpu': 'fixture'}
    module.on_workspace_updated(workspace=state)
    assert len(saves) == 1
    state['recent_files'] = ['new.py']
    module.on_workspace_updated(workspace=state)
    assert len(saves) == 2
    kernel.shutdown()


def test_workspace_failed_save_is_retryable(tmp_path):
    from core import Kernel
    from core.workspace_runtime import WorkspaceRuntimeModule
    kernel = Kernel()
    module = WorkspaceRuntimeModule(kernel, state_path=tmp_path/'workspace.json')
    state = kernel.workspace_manager.snapshot()
    calls = []
    def save(state):
        calls.append(state)
        if len(calls) == 1:
            raise OSError('fixture')
    module.state_store.save = save
    module.on_workspace_updated(workspace=state)
    module.on_workspace_updated(workspace=state)
    assert len(calls) == 2
    kernel.shutdown()


@pytest.fixture
def log_environment(monkeypatch):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    for name in ('ASTA_LOG_FILE', 'ASTA_LOG_FORMAT', 'ASTA_LOG_LEVEL'):
        monkeypatch.delenv(name, raising=False)
    yield root
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_logging_config_is_idempotent_and_preserves_foreign_handlers(log_environment):
    from core.logging_setup import configure_logging
    root = log_environment
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        old = configure_logging()
        new = configure_logging()
        assert foreign in root.handlers
        assert all(h not in root.handlers for h in old)
        assert all(h in root.handlers for h in new)
        assert sum(getattr(h, '_asta_owned', False) for h in root.handlers) == 1
    finally:
        root.removeHandler(foreign)


def test_logging_json_file_rotation_and_level(log_environment, monkeypatch, tmp_path):
    from core.logging_setup import configure_logging
    from logging.handlers import RotatingFileHandler
    path = tmp_path/'runtime.log'
    monkeypatch.setenv('ASTA_LOG_FILE', str(path))
    monkeypatch.setenv('ASTA_LOG_FORMAT', 'json')
    monkeypatch.setenv('ASTA_LOG_LEVEL', 'WARNING')
    handlers = configure_logging()
    file_handler = next(h for h in handlers if isinstance(h, RotatingFileHandler))
    file_handler.maxBytes = 200
    logger = logging.getLogger('asta.fixture')
    logger.info('hidden message')
    for i in range(8):
        logger.warning('fixture message %s', i)
    assert (tmp_path/'runtime.log.1').exists()
    data = json.loads(path.read_text().splitlines()[-1])
    assert data['logger'] == 'asta.fixture' and data['level'] == 'WARNING'
    assert 'hidden message' not in path.read_text()


def test_invalid_log_level_fails_without_replacing_handlers(log_environment, monkeypatch):
    from core.logging_setup import configure_logging
    before = list(log_environment.handlers)
    monkeypatch.setenv('ASTA_LOG_LEVEL', 'not-a-level')
    with pytest.raises(ValueError):
        configure_logging()
    assert log_environment.handlers == before


@pytest.mark.parametrize('module', ['ai/ai_module.py', 'voice/voice_module.py',
    'speech/speech_module.py', 'memory/memory_module.py', 'core/tools/module.py'])
def test_hot_path_modules_use_logging_not_print_monkeypatches(module):
    tree = ast.parse((ROOT/module).read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'print' for n in ast.walk(tree))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'info','warning'} for n in ast.walk(tree))


def test_packaging_extras_and_compatibility_files_are_canonical():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads((ROOT/'pyproject.toml').read_text())
    extras = config['project']['optional-dependencies']
    assert set(extras) == {'kokoro', 'browser', 'laya', 'mempalace', 'indic', 'wakeword-training', 'dev'}
    for extra in ('kokoro','browser','laya','mempalace'):
        lines = [l for l in (ROOT/f'requirements-{extra}.txt').read_text().splitlines() if l and not l.startswith('#')]
        assert lines == [f'-e .[{extra}]']
    assert '-e .[indic]' in (ROOT/'voice/requirements-indic.txt').read_text()
    assert '-e .' in (ROOT/'requirements.txt').read_text()
    assert config['project']['requires-python'] == '>=3.10'


@pytest.mark.parametrize('match,throws', [(False, False), (True, False), (False, True)])
def test_isolated_face_experiment_releases_camera(monkeypatch, tmp_path, match, throws):
    from experiments.face_identity import FaceRecognizer
    calls = []
    def read():
        if throws:
            raise KeyboardInterrupt()
        return True, object()
    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: None, read=read,
        release=lambda: calls.append('release'))
    cv = SimpleNamespace(VideoCapture=lambda *a: camera, CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4, COLOR_BGR2RGB=1, flip=lambda *a: object(), cvtColor=lambda *a: object())
    face = SimpleNamespace(face_locations=lambda rgb: [], face_encodings=lambda *a: [np.zeros(128)],
        compare_faces=lambda *a: [match])
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    monkeypatch.setitem(sys.modules, 'face_recognition', face)
    path = tmp_path/'embeddings.npy'
    np.save(path, np.zeros((1,128)))
    recognizer = FaceRecognizer(path)
    if throws:
        with pytest.raises(KeyboardInterrupt):
            recognizer.face_reco(max_frames=2)
    else:
        assert recognizer.face_reco(max_frames=2) is match
    assert calls == ['release']


def test_face_experiment_never_unpickles_or_imports_in_runtime():
    source = (ROOT/'experiments/face_identity.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    assert 'allow_pickle=False' in source
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and 'pickle' in ast.unparse(n) for n in ast.walk(tree))
    assert not (ROOT/'vision/face_recognition.py').exists()


def test_removed_duplicates_have_no_runtime_imports():
    removed = ['recognition_engine_2', 'polly_engine', 'open_last_screenshot', 'open_last_screenshot_tool']
    for folder in ('ai','core','voice','speech','vision','hud','memory'):
        for path in (ROOT/folder).rglob('*.py'):
            if 'openWakeWord' in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    assert not any(name in ast.unparse(node) for name in removed), path


def test_repository_audit_is_read_only():
    path = ROOT/'scripts/audit_repository.py'
    spec = importlib.util.spec_from_file_location('audit_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = subprocess.check_output(['git','status','--porcelain'], cwd=ROOT)
    findings = module.audit(ROOT)
    after = subprocess.check_output(['git','status','--porcelain'], cwd=ROOT)
    assert before == after
    assert all(set(item) == {'path','category'} for item in findings)


def test_ui_search_fallback_uses_literal_clipboard_not_escaped_sendkeys():
    from core.tools import ui_play
    assert '[System.Windows.Forms.Clipboard]::SetText($env:ASTA_UI_QUERY)' in ui_play._SEARCH_IN_APP
    assert "SendWait('^v')" in ui_play._SEARCH_IN_APP
    assert 'SendWait($env:ASTA_UI_KEYS)' not in ui_play._SEARCH_IN_APP


def test_post_action_application_verification_invalidates_cached_process_state():
    from core.autonomy.verification import VerificationEngine
    calls = []
    manager = SimpleNamespace(invalidate_process_cache=lambda: calls.append('invalidate'),
        is_application_running=lambda query: calls.append('check') or True)
    engine = VerificationEngine(application_manager=manager)
    task = SimpleNamespace(id='task')
    step = SimpleNamespace(id='step', metadata={'target':'fixture'}, arguments={'target':'fixture'})
    result = SimpleNamespace(output={'target':'fixture'})
    verification = engine._verify_application_running(task, step, result)
    assert verification.status.value == 'verified'
    assert calls == ['invalidate', 'check']
