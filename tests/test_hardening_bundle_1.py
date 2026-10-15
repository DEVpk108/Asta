"""Native hook ownership, JSON boundaries, pixel parity and clipboard safety."""
import ast
import importlib
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace
import zlib

import pytest

from ai.ai_module import AIModule
from ai.runtime_patch import apply_ai_runtime_patch
from ai.response_formatter import _verified_visual_response
from core import Kernel
from core.agent.brain import AgentBrain
from core.agent.intent_resolver import parse_resolution
from core.background_tasks import BackgroundTaskCoordinator
from core.computer.controller import ComputerController, ComputerControlError
from core.contracts import IntentResult, IntentType, ToolRequest, ToolResult
from core.tools import ToolRequestBuilder
from vision.screenshot_backend import _encode_rgb_png


def make_ai():
    kernel=Kernel()
    ai=AIModule.__new__(AIModule)
    ai.kernel,ai.event_bus=kernel,kernel.event_bus
    ai.tool_request_builder=ToolRequestBuilder(kernel.tool_registry)
    ai.chat_history=SimpleNamespace(latest_active_context=lambda **kwargs: [])
    ai.engine=SimpleNamespace(generate_response=lambda text, **kwargs: 'Chat response.')
    replies=[];ai._emit_assistant_text=replies.append
    return ai,kernel,replies


def test_all_legacy_hooks_and_package_reload_leave_native_methods_unchanged():
    names=('on_user_message','on_tool_result','_is_creator_identity_question','_format_tool_success','_generate_response')
    original={name:AIModule.__dict__[name] for name in names}
    import ai
    importlib.reload(ai)
    for _ in range(3): apply_ai_runtime_patch()
    assert all(AIModule.__dict__[name] is value for name,value in original.items())
    tree=ast.parse(Path('main.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Name) and n.id == 'apply_ai_runtime_patch' for n in ast.walk(tree))


@pytest.mark.parametrize('text', ['who created you','who made asta','Who programmed you?',
    'Who designed A.S.T.A.?','who is your developer','who is behind asta'])
def test_native_creator_identity_covers_existing_variants_without_hooks(text):
    ai,kernel,replies=make_ai()
    try:
        ai.on_user_message(text)
        assert replies == ['I was created by Mr. PRASANT KUMAR.']
    finally: kernel.shutdown()


@pytest.mark.parametrize('text', ['Who created you?', 'Open the screenshot', 'Okay, take a screenshot'])
def test_native_shortcuts_do_not_bypass_pending_consent(text):
    ai,kernel,replies=make_ai()
    request=ToolRequest(tool='system.run_command', arguments={},request_id='pending')
    kernel.approval_manager.request_approval(request,'needs consent')
    decisions=[]
    kernel.event_bus.subscribe('tool_confirmation_response',lambda **data: decisions.append(data))
    ai.tool_request_builder=SimpleNamespace(build=lambda intent: ToolRequest(tool='vision.screenshot',arguments={},request_id='shot'))
    try:
        ai.on_user_message(text)
        assert decisions == [{'request_id':'pending','approved':False}]
    finally: kernel.shutdown()


@pytest.mark.parametrize('text, action, key', [
    ('Open the screenshot','open_screenshot','screenshot_open_request'),
    ('show latest screen shot','open_screenshot','screenshot_open_request'),
    ('Okay, take a screenshot','screenshot','direct_screenshot_request'),
    ('grab a screen capture','screenshot','direct_screenshot_request')])
def test_native_screenshot_shortcuts_dispatch_without_hook(text,action,key):
    ai,kernel,_=make_ai();seen=[];built=[]
    ai.tool_request_builder=SimpleNamespace(build=lambda intent: built.append(intent) or ToolRequest(tool='vision.screenshot',arguments={},request_id='shot'))
    kernel.event_bus.subscribe('tool_request',lambda request:seen.append(request))
    try:
        ai.on_user_message(text)
        assert built[0].entities['action'] == action
        assert len(seen) == 1 and seen[0].metadata[key] is True
    finally: kernel.shutdown()


@pytest.mark.parametrize('text', ['Okay, turn off conversation mode.', 'Please, turn on conversation mode.'])
def test_native_conversation_filler_handling_without_hook(text):
    ai,kernel,replies=make_ai();modes=[]
    kernel.event_bus.subscribe('conversation_mode_set',lambda enabled:modes.append(enabled))
    try:
        ai.on_user_message(text)
        assert modes == [('off' not in text)] and len(replies) == 1
    finally: kernel.shutdown()


@pytest.mark.parametrize('verified', ['false','true',1,[],{},False,None])
def test_final_visual_wording_requires_real_boolean_verdict(verified):
    result=ToolResult(tool='vision.inspect',success=True,output={'verified':verified,'summary':'claimed result'})
    assert _verified_visual_response(result) is None
    assert 'is confirmed' not in AIModule._format_tool_success(result)


def test_final_visual_wording_respects_negative_visual_match():
    result=ToolResult(tool='vision.inspect',success=True,output={'verified':True,'visual_match':False})
    assert _verified_visual_response(result) is None
    result=ToolResult(tool='vision.inspect',success=True,output={'verified':True,'visual_match':True,'summary':'The window is open'})
    assert _verified_visual_response(result) == 'Done — The window is open.'


def test_native_result_suppression_for_retry_and_intermediate_steps():
    ai,kernel,replies=make_ai()
    step1=SimpleNamespace(id='first',status=SimpleNamespace(value='completed'))
    step2=SimpleNamespace(id='last',status=SimpleNamespace(value='completed'))
    plan=SimpleNamespace(steps=[step1,step2],get_step=lambda step_id:step1)
    task=SimpleNamespace(plan=plan,metadata={})
    kernel.task_manager=SimpleNamespace(get=lambda task_id:task,current=lambda:task)
    ai.on_tool_result(ToolResult(tool='test',success=False,error='old failure',metadata={'task_id':'task','plan_step_id':'first'}))
    ai.on_tool_result(ToolResult(tool='test',success=True,output='intermediate',metadata={'task_id':'task','plan_step_id':'first'}))
    assert not replies
    ai.on_tool_result(ToolResult(tool='test',success=True,output='final',metadata={'task_id':'task','plan_step_id':'last'}))
    assert replies == ['final']


@pytest.mark.parametrize('raw, query', [
    ('prefix {"context":"some {braces}"} next {"action":"play", "query":"one"}', 'one'),
    ('```json\n{"action":"play","query":"a } b"}\n``` trailing {"other":1}', 'a } b'),
    ('{"bad": } then {"action":"play", "query":"two"}', 'two'),
    ('<think>{"action":"play", "query":"wrong"}</think>{"action":"play", "query":"correct"}', 'correct'),
    ('{"action":"play","query":"first"} {"action":"play","query":"second"}', 'first')])
def test_resolution_decodes_json_boundaries_not_greedy_brace_span(raw,query):
    result=parse_resolution(raw)
    assert result is not None and result.action == 'play' and result.query == query


@pytest.mark.parametrize('raw', ['not JSON','{"action":"unknown"}','{"context":"x"}', '{"action":'])
def test_invalid_resolution_still_rejected(raw):
    assert parse_resolution(raw) is None


@pytest.mark.parametrize('raw, expected', [
    ('{"items":["apple","banana', {'items':['apple','banana']}),
    ('{"items":["apple","banana"', {'items':['apple','banana']}),
    ('{"items":["a\\\"b","c', {'items':['a"b','c']}),
    ('{"items":[1,2,', {'items':[1,2]}),
    ('{"valid":1,"dangling', {'valid':1}),
    ('{"valid":1,"dangling":', {'valid':1})])
def test_json_repair_does_not_delete_last_array_string(raw,expected):
    assert AgentBrain._close_truncated_json(raw) == expected


def decode_png(png):
    chunks=[];offset=8
    while offset < len(png):
        length=struct.unpack('>I',png[offset:offset+4])[0]
        kind=png[offset+4:offset+8];payload=png[offset+8:offset+8+length]
        crc=struct.unpack('>I',png[offset+8+length:offset+12+length])[0]
        assert zlib.crc32(kind+payload)&0xffffffff == crc
        chunks.append((kind,payload));offset+=length+12
    return zlib.decompress(b''.join(payload for kind,payload in chunks if kind == b'IDAT'))


@pytest.mark.parametrize('width,height', [(1,1),(2,3),(32,17),(511,5),(1280,720)])
def test_fast_png_preserves_all_rgb_pixels_and_scanline_boundaries(width,height):
    size=width*height*4;bgra=(bytes(range(256))*((size+255)//256))[:size]
    rows=decode_png(_encode_rgb_png(width,height,bgra))
    for y in range(height):
        source=bgra[y*width*4:(y+1)*width*4]
        expected=b''.join(bytes((source[x+2],source[x+1],source[x])) for x in range(0,len(source),4))
        assert rows[y*(width*3+1):(y+1)*(width*3+1)] == b'\x00'+expected


@pytest.mark.parametrize('width,height,bgra', [(0,1,b''),(1,0,b''),(-1,1,b''),(2,2,b'wrong')])
def test_invalid_png_inputs_rejected(width,height,bgra):
    with pytest.raises(ValueError): _encode_rgb_png(width,height,bgra)


class Clipboard:
    def __init__(self): self.value='previous';self.copies=[]
    def copy(self,value): self.value=value;self.copies.append(value)
    def paste(self): return self.value


def test_async_paste_reads_requested_value_after_method_returns():
    clipboard=Clipboard();queued=[]
    backend=SimpleNamespace(hotkey=lambda *keys:queued.append(lambda:clipboard.paste()))
    result=ComputerController(backend=backend,clipboard=clipboard).type_text('new payload')
    assert result['method'] == 'clipboard_paste' and result['clipboard_restored'] is False
    assert queued[0]() == 'new payload' and clipboard.copies == ['new payload']


def test_clipboard_copy_failure_can_fall_back_before_paste_is_issued():
    calls=[];clipboard=Clipboard()
    clipboard.copy=lambda value:(_ for _ in ()).throw(RuntimeError('unavailable'))
    backend=SimpleNamespace(write=lambda value,interval:calls.append(value))
    result=ComputerController(backend=backend,clipboard=clipboard).type_text('hello')
    assert result['method'] == 'direct_typing' and calls == ['hello']


def test_hotkey_failure_never_falls_back_to_double_text_entry():
    calls=[];clipboard=Clipboard()
    def hotkey(*keys): raise RuntimeError('OS may have queued input')
    backend=SimpleNamespace(hotkey=hotkey,write=lambda *args,**kwargs:calls.append(args))
    with pytest.raises(ComputerControlError,match='not retrying'):
        ComputerController(backend=backend,clipboard=clipboard).type_text('hello')
    assert not calls and clipboard.value == 'hello'


def voice_methods():
    tree=ast.parse(Path('voice/voice_module.py').read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='VoiceModule')
    methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in {'_can_listen','on_conversation_mode_set'}]
    namespace={'time':SimpleNamespace(monotonic=lambda:10.), 'logger':__import__('logging').getLogger('voice.voice_module')}
    exec(compile(ast.Module(body=methods,type_ignores=[]),'<native-voice-owners>','exec'),namespace)
    return namespace


def test_native_voice_cooldown_preserves_existing_listening_guards():
    methods=voice_methods();voice=SimpleNamespace(_running=True,_tts_active=False,_tts_guard_until=0.,_conversation_active=False,
        microphone=SimpleNamespace(clear_buffer=lambda:None))
    methods['on_conversation_mode_set'](voice,False)
    assert voice._asta_wakeword_cooldown_until == 12.
    assert methods['_can_listen'](voice) is False
    methods['time'].monotonic=lambda:12.
    assert methods['_can_listen'](voice) is True
    voice._tts_active=True
    assert methods['_can_listen'](voice) is False
    voice._tts_active=False;voice._running=False
    assert methods['_can_listen'](voice) is False


def test_background_snapshot_cleared_during_start_does_not_crash():
    coordinator=BackgroundTaskCoordinator.__new__(BackgroundTaskCoordinator)
    coordinator.cancelled=threading.Event();phases=[]
    coordinator._phase=lambda phase,message:phases.append(phase)
    coordinator.kernel=SimpleNamespace(task_runtime=SimpleNamespace(start_goal=lambda goal,intent:SimpleNamespace(status=SimpleNamespace(value='active'))))
    coordinator.snapshot=lambda:None
    coordinator._start_goal('goal',None)
    assert phases == ['planning','working']


def test_decision_prompt_does_not_build_unused_definition_payload():
    source=Path('core/agent/brain.py').read_text(encoding='utf-8');tree=ast.parse(source)
    method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='_build_decision_prompt')
    assert not any(isinstance(node,ast.Name) and node.id=='definitions' for node in ast.walk(method))
