"""Compatibility hook for A.S.T.A.'s low-latency speech interruption.

The live VoiceModule now performs acoustic speech-onset detection and hands the
captured onset to the normal VAD/Whisper command path. This module is retained
for older startup code that may still call apply_voice_interrupt_patch().
"""

from .voice_module import VoiceModule


def apply_voice_interrupt_patch():
    """Keep compatibility with older startup code without replacing the
    low-latency VoiceModule barge-in implementation.
    """
    return VoiceModule._barge_listen_loop
