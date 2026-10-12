# Dialogue manager: turning conversation into well-defined project tasks

Status: design, not implemented yet.
Branch: `feat/edit-fix-loop` (implemented on a new branch `feat/dialogue-manager`).

## 1. Problem

ASTA judges every sentence on its own. Six separate pieces each guess what a
sentence means:

| Piece | Where | Looks at |
|---|---|---|
| Rule router | `core/intent_router.py` | one sentence, fixed patterns |
| Project follow-up patterns | `core/agent/project_followup.py` | one sentence + last task (in memory only) |
| Project conversation check | `core/agent/project_conversation.py` | one sentence + a few turns, model decides *act or ask* |
| "What did you mean" resolver | `core/agent/intent_resolver.py` | one sentence + last task, media/apps only |
| Fast intent model (System 1, Laya) | `ai/ai_module.py` | one sentence |
| Chat model | `ai/llama_cpp_engine.py` | the conversation, but invents tool calls it may not make |

None of them keeps the state of the conversation ("we're discussing
`snake.py`, the user wants a change, we don't know which one yet"). So every
new phrasing needs another rule. When a model makes the call instead, it is
a 2.6B model, and small models choose *act* over *ask* (live run: "make
changes in the snake game" became the task "make appropriate changes").

The failure then went down the chain:

* the coding planner got a goal with nothing to do and invented a plan;
* its retry pasted all of `snake.py` back into its reply, ran out of output
  tokens and failed;
* the user was told to edit `.env`.

## 2. Goals and non-goals

Goals

1. A request is built up over several turns until it is clear enough to act on.
   One component owns this state.
2. ASTA asks only about what is actually missing, in natural words, and never
   asks again for something the user already said.
3. ASTA never starts a project task from a guess. Inferred tasks are repeated
   back in one sentence first; explicit ones start straight away.
4. Project context survives a restart ("the snake game we were building").
5. Every decision can be tested offline against the real local model, so a
   change can be measured instead of judged by one live run.

Non-goals (this slice)

* Replacing the fast rules for unambiguous commands (media, open/close apps,
  notes, screenshots). They stay; they are fast and correct.
* Resuming interrupted tasks after a restart (see
  `project-workspace-continuity.md`: no plan, approval or task replay).
* Changing approvals, the sandbox or verification. A finished draft starts
  an ordinary project task.

## 3. Architecture

```
user turn
   |
   v
[fast rules] --unambiguous command--> command path (unchanged)
   |
   v otherwise
[DialogueManager]
   1. understand: model fills/updates a TaskDraft   (structured output)
   2. policy:     plain code decides                (deterministic)
   3. act:        chat | ask | confirm | start | cancel
   |
   +--start--> planner --> (plan | needs_clarification) --> task runtime
```

New package `core/dialogue/`:

* `state.py`: `DialogueState` for each chat session: the current `TaskDraft`,
  the question waiting for an answer (and which field it is about), the last
  task, and when each was set. In memory, keyed by chat session id.
* `draft.py`: `TaskDraft` and its completeness rules (section 4).
* `understand.py`: the one model call (section 5).
* `policy.py`: draft + state -> action (section 6).
* `journal.py`: the project journal (section 8).
* `manager.py`: `DialogueManager.handle(text) -> Action`, called from
  `AIModule.on_user_message` after the fast rules and before chat.

Removed when this lands: `core/agent/project_followup.py`,
`core/agent/project_conversation.py`, the project part of
`_resolve_contextual_intent`, and the chat model's project tool-call fallback
(the chat model is never asked to act; it only chats).

## 4. The task draft

```
TaskDraft
  kind:        "change" | "create" | "fix" | "run" | "explain" | None
  project:     workspace name (always known: from the workspace manager)
  files:       [paths]   existing files the change is about (or new file names)
  change:      str       what should be different, in the user's words
  details:     [str]     concrete details given (keys, names, numbers, colours)
  check:       "open" | "run_tests" | "run" | None   how the user will see it
  source:      "explicit" | "inferred"   did the user say it, or did we infer it
  turns:       [turn ids that contributed]
```

Completeness rules (plain code, not the model):

| Field | Complete when |
|---|---|
| `files` | each path exists in the workspace listing, **or** `kind == "create"` with a file name; if one file in the project fits (e.g. "the snake game" and only `snake.py`), it is filled and marked inferred |
| `change` | has an action verb **and** a concrete object ("add W/A/S/D controls", "make the snake faster"); vague words alone ("changes", "improvements", "appropriate", "something", "stuff") do not count |
| `check` | optional; defaults to `open` for desktop programs, `run_tests` when tests exist for the file, else none |

A draft is **ready** when `files` and `change` are complete.

## 5. Understanding (the only model call)

Input (compact, about 500 tokens):

* project name and file listing (at most 30 paths);
* the journal's last 3 entries (section 8);
* the last 6 chat turns;
* the current draft as JSON, and the pending question, if any;
* the new user sentence.

Output: a JSON patch to the draft plus one intent label, forced by a JSON
schema (llama.cpp `json_schema`), so the output always parses:

```json
{
  "about_project": true,
  "intent": "new_request | adds_detail | answers_question | confirms | rejects | cancels | unrelated",
  "kind": "change",
  "files": ["snake.py"],
  "change": "move with W, A, S and D as well as the arrow keys",
  "details": ["W", "A", "S", "D"],
  "check": "open"
}
```

The model is asked to **extract only what the user said**, never to decide
whether to act. Small models are good at extracting and poor at deciding,
so this uses them where they are strong. Empty fields stay empty; the prompt
gives short worked examples of vague requests left empty.

Cost: one call of at most 1 second, made only when the fast rules didn't
handle the turn and either a draft is open or the turn mentions the project
or code (the current word gate). Small talk skips it.

Which model: the general model by default. The test set (section 9) decides
whether the coding model (Qwen3.5-4B) or a single resident 7-8B model is worth
the memory or the swap time. See open questions.

## 6. Policy (deterministic)

| Situation | Action |
|---|---|
| `about_project` false, no draft open | chat as today |
| `cancels` | drop the draft; "Okay, I'll leave it." |
| draft not ready | **ask** about the first missing field, the question generated from a template **plus** the model's wording of the user's own words ("What would you like to change in snake.py?"); remember which field was asked |
| draft ready, `source == explicit` (user named file and change in this turn) | **start** |
| draft ready, anything inferred | **confirm** in one sentence: "I'll make the snake also move with W, A, S and D in snake.py and reopen it. Go ahead?" |
| pending confirm + `confirms` | **start** |
| pending confirm + `adds_detail` | update the draft, confirm again (once) |
| pending confirm + `rejects` | ask what to change instead |
| same field asked twice without progress | stop asking, explain what's missing, offer an example built from the project's files |
| draft older than 10 minutes, or unrelated turn | drop the draft quietly (a pending question expires) |

Start = build the planner goal from the draft
(`"In snake.py: make the snake also move with W, A, S and D. Keep the arrow keys working. Then open snake.py on the desktop."`)
and hand it to the existing background task path.

## 7. Planner safety net

1. The planner output may be `{"needs_clarification": "<question>"}` instead of
   steps. The dialogue manager turns that into an **ask** and keeps the draft.
   No task is created.
2. ASTA never makes the model copy file text: a verified readback of a file
   the plan did not write gets `expected_content` filled from disk by ASTA
   (it already does this for written files). This removes the failure that
   cut the live run off.
3. User-facing failure text never names settings. Developer detail stays in
   the console and the HUD's details panel.

## 8. Project journal (survives restarts)

`<project>/.asta/journal.jsonl`, one line per finished task:

```json
{"at": "2026-10-12T15:20:00Z", "goal": "Write a Snake game ...", "files": ["snake.py"],
 "outcome": "completed", "summary": "Wrote snake.py (83 lines), verified, opened on the desktop"}
```

* Written by ASTA on task end; last 50 entries kept; `.asta/` added to the
  project's `.gitignore` when the project uses git.
* Holds goals, file names and outcomes only. No file contents, plans,
  approvals or environment.
* Read into the understanding prompt (last 3 entries) together with
  `git log -5 --oneline` when the project is a git repository.
* Context only. Nothing is resumed or replayed.

## 9. Conversation test set

`tests/dialogues/*.yaml`, one conversation per file:

```yaml
name: vague request after restart, then the detail
project: {files: ["snake.py"], journal: ["Write a Snake game (completed)"]}
turns:
  - user: "Hey I want you to make changes in the snake game we were building"
    expect: {action: ask, about_field: change, mentions: ["snake.py"]}
  - user: "I'd like it to move with W, A, S, D too"
    expect: {action: confirm, draft: {files: ["snake.py"], details_include: ["W", "A", "S", "D"]}}
  - user: "yes go ahead"
    expect: {action: start, goal_includes: ["snake.py", "W"]}
```

Two ways to run it:

* **Offline in CI** (`pytest`): the model is replaced by recorded answers;
  this tests the policy and the plumbing.
* **Against the real model** (`python -m tools.dialogue_eval --model general`):
  runs every conversation on the local llama.cpp server and prints a score
  per expected field, and the confusion between ask, confirm, start and chat.
  Run it before and after every prompt or model change.

Target before switching it on by default: **0** wrong *start* actions (never
act on a guess), **at least 90%** correct *ask*/*confirm*/*chat* actions, median
added latency **at most 1 s**.

Initial set (about 40 conversations): vague requests, details over several turns,
pronouns ("make it faster"), answers to ASTA's questions, cancels and
"never mind", unrelated chat in between, music during a draft, misheard words
(speech recognition), Hinglish, explicit one-shot requests (must start without
confirming), questions about code (must chat, not act).

## 10. Phases

| Phase | Delivers | Done when |
|---|---|---|
| 1 | Test set format + 40 conversations + offline runner + real-model runner | runner prints a baseline for today's code |
| 2 | `core/dialogue` (state, draft, understand, policy) behind `ASTA_DIALOGUE_MANAGER=1`; the two follow-up modules removed | offline tests pass; real-model score meets the targets above |
| 3 | Planner safety net (section 7) | the live "make changes in the snake game" run asks instead of failing |
| 4 | Project journal | the restart conversation in section 9 passes live |
| 5 | Switch on by default; delete the old paths | one week of normal use without a wrong start |

## 11. Open questions

1. **GPU memory.** With 12 GB or more, one resident 7-8B model for
   understanding and planning may beat swapping between 2.6B and 4B. The test
   set measures it. Until then the default is the general model.
2. **Confirm policy.** Assumed: confirm only when something was inferred;
   explicit one-shot requests start directly.
3. **Latency.** Assumed: up to 1 s extra on project-related turns is fine.
4. **The last two patches** (follow-up patterns, project conversation) stay
   on `feat/edit-fix-loop` for now and are removed in phase 2, so nothing
   working is lost before the replacement scores better.
