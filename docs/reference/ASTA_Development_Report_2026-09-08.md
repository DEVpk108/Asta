# ASTA Development Report 2026-09-08

Converted reference from `ASTA_Development_Report_2026-09-08.docx`. The original document is retained.

A.S.T.A. Development & Optimization Report

Local AI Runtime, Model Lifecycle, Latency & Voice Pipeline

Session date: 8 September 2026

Executive Summary

Today’s work focused on making A.S.T.A. faster, more stable, and safer to run repeatedly during development. We traced the AI response path from the Python client through LM Studio and the streaming SSE layer, fixed repeated model-instance creation, removed a roughly two-second local HTTP startup delay, and verified that the local speech pipeline was also producing much lower first-audio latency after warm-up.

The final user-run regression suite completed with 36 passing tests. The final runtime logs show HTTP response headers arriving in roughly 19–31 ms instead of about 2 seconds, client TTFT around 175–195 ms, and LM Studio generation around 68–72 tokens/s on several requests.

## 1. Starting Problems

| Issue | Impact |
| --- | --- |
| Repeated LM Studio model loading | Each A.S.T.A. test run caused another instance of the same Nemotron model to appear in LM Studio (for example, the second instance was shown with a :2 suffix). |
| High client-side TTFT despite fast model generation | Earlier measurements showed roughly 2.0–2.1 s before A.S.T.A. received HTTP headers even though LM Studio reported roughly 0.1–0.2 s TTFT. |
| Need for safe repeatable warm-up | The warm-up path was intentionally added to remove first-request model-loading work, but its explicit load request was creating a new model instance every startup. |
| Speech startup/first-audio variability | Kokoro was warm-loaded and later actual inference warm-up was added; this was checked as part of the overall voice-latency path. |

## 2. Step-by-Step Work Completed

Step 1 - Instrument the AI request path

A.S.T.A.’s LM Studio engine was instrumented to distinguish client TTFT from LM Studio’s own reported TTFT. The code also records HTTP-header arrival, first SSE event, chat.start, prompt-processing duration, message-start timing, and model-load timing. This separated model-generation time from transport/setup time.

Step 2 - Add explicit AI warm-up

An AIEngine.warmup() path was introduced to load the selected model and perform a tiny non-stored inference using "Ready." with one output token. This was intended to move model initialization away from the first user interaction.

Step 3 - Diagnose the repeated-instance bug

The explicit /api/v1/models/load call was identified as the reason repeated A.S.T.A. startups could create additional instances. The fix was to query the loaded model list first and reuse an existing loaded instance when one already exists.

Step 4 - Make model reuse explicit

A.S.T.A. now checks GET /api/v1/models, matches the configured model key, extracts an existing loaded instance ID, and stores that ID in model_instance_id. A new model load is performed only when no matching instance exists.

Step 5 - Add regression tests for model lifecycle

Tests were expanded to cover both paths: loading when the model is absent and reusing an existing instance without calling /models/load again. This increased the local suite to 36 passing tests.

Step 6 - Investigate the ~2-second HTTP delay

The timing data showed that LM Studio’s reported TTFT was already low while A.S.T.A. was waiting roughly two seconds before HTTP headers arrived. The bottleneck therefore appeared to be local client transport/setup rather than generation.

Step 7 - Optimize loopback transport

The AI engine was changed to use 127.0.0.1 for local LM Studio communication and to bypass inherited environment proxy settings for loopback requests by disabling requests.Session trust of environment proxy configuration.

Step 8 - Verify the latency fix

After the transport change, the user’s runtime logs showed HTTP headers in 0.019–0.031 s and client TTFT in 0.175–0.195 s. This confirmed that the large pre-SSE delay had effectively been removed.

Step 9 - Verify the local speech path

The runtime also showed Kokoro first-audio latency around 0.050–0.107 s for several responses in the final run, confirming that the warmed local TTS path was behaving well during the same end-to-end test.

## 3. Final Architecture After Today’s Fixes

A.S.T.A.’s relevant runtime path is now:

User voice input → Voice/VAD → AI module → AIEngine → local LM Studio (127.0.0.1) → SSE stream → sentence callback → Kokoro TTS → playback.  At startup, AIEngine first checks loaded model instances and reuses an existing instance before considering a new load.

The transport path also avoids proxy/environment handling for loopback traffic, eliminating the large local HTTP setup delay observed earlier.

## 4. Final Validation Results

| Check | Final observation | Status |
| --- | --- | --- |
| Automated tests | 36 passed in 7.95 s | PASS |
| Loaded-model lookup | 0.005 s in final run | PASS |
| Model instance reuse | Existing nvidia/nemotron-3-nano-4b:2 reused; no new instance created | PASS |
| Inference warm-up | 0.082 s in final run | PASS |
| HTTP headers | 0.019–0.031 s | PASS |
| Client TTFT | 0.175–0.195 s on sampled conversational requests | PASS |
| LM Studio reported TTFT | 0.090–0.150 s on sampled requests | PASS |
| Generation speed | 56.6–71.8 tok/s in sampled requests | PASS |
| Kokoro TTFA | 0.050–0.107 s on several final responses | PASS |

## 5. What the Logs Prove

- Model lifecycle is stable: the startup check finds an existing loaded instance and reuses it.

- The large client-side transport delay is gone: HTTP headers now arrive in tens of milliseconds.

- LM Studio generation itself was never the main problem in the final state; model TTFT remained around one tenth of a second and generation stayed near 70 tok/s.

- The local TTS path is warmed and producing first audio quickly in the final run.

- The existing tool runtime remains active: screenshot worked end-to-end, while medium-risk application opening correctly triggered the approval policy.

## 6. Remaining Known Issue for the Next Session

The final test exposed a separate functional issue in the tool-approval conversation path. When A.S.T.A. said it needed confirmation before opening Chrome, the follow-up user response "I confirm." was classified as unknown and went through normal conversation generation instead of completing the pending tool confirmation. This is separate from today’s latency work and should be handled as a tool-runtime / approval-state bug next.

## 7. Git / Change Record

Work was performed on the feature/tool-runtime-integration branch. The session included the model-instance reuse change, the direct loopback transport optimization, and corresponding regression coverage. The user pulled the latest branch and ran the full test suite successfully.

Final user pull reached commit 4c4c760, and the user’s local test run reported 36 passed in 7.95 seconds.

## 8. Session Conclusion

Today’s optimization work is complete for the current AI transport and model lifecycle scope. A.S.T.A. now starts without accumulating duplicate Nemotron instances, responds through the local LM Studio path with sub-200 ms client TTFT in the final tests, and retains the modular tool/speech architecture. The next development task is the pending confirmation flow for medium-risk tools.

Source: Final A.S.T.A. terminal/test log supplied in the conversation; numerical validation values above are taken directly from that log.
