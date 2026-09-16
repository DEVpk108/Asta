# A.S.T.A. + MemPalace

This is the first implementation step of A.S.T.A.'s long-term Memory Layer.

## What we are building

A.S.T.A. keeps its existing SQLite chat history for the current conversation.
MemPalace is added as a separate long-term memory backend.

```text
User message
    |
    v
A.S.T.A. Kernel
    |
    +--> Chat History (current conversation)
    |
    +--> MemoryModule
            |
            +--> recall relevant memories
            |
            +--> store the new message
            |
            v
        MemPalace
            |
            +--> semantic memory search
            +--> layered memory stack
            +--> local knowledge graph (future integration)
```

## Important design decision

A.S.T.A. does not depend directly on MemPalace throughout the codebase.
`memory/mempalace_adapter.py` is the integration boundary.

This means MemPalace can later be replaced, combined with another backend, or
used alongside additional memory stores without rewriting the kernel.

## Current integration

The runtime now has a `MemoryModule` which:

- initializes MemPalace when the package is installed;
- recalls relevant memories before the AI handles a new user message;
- stores user messages as verbatim conversation memories;
- stores assistant sentences as verbatim conversation memories;
- exposes the retrieved context to the AI through an A.S.T.A.-owned runtime bridge;
- fails gracefully when MemPalace is not installed or cannot open its palace.

The current MemPalace wing is `asta` and conversation memories are stored in
the `conversation` room.

## Installation

From the A.S.T.A. project virtual environment:

```powershell
pip install -r requirements-mempalace.txt
```

Optional palace location:

```powershell
$env:ASTA_MEMPALACE_PATH = "E:\AstaMemory\palace"
```

When no path is configured, A.S.T.A. uses MemPalace's configured default
palace path.

## Current memory flow

For a new message:

```text
User: "What was that touch panel idea we discussed?"
          |
          v
MemoryModule searches MemPalace
          |
          v
Relevant memories are attached as context
          |
          v
A.S.T.A. AI generates the response
          |
          v
The new user/assistant exchange is stored
```

Retrieved memory is treated as fallible context. It is not treated as an
instruction and it does not override the current user request or A.S.T.A.'s
system rules.

## What is deliberately not implemented yet

This is not the final A.S.T.A. Memory Layer.

Future stages will separate and connect:

- working memory;
- episodic memory;
- semantic knowledge;
- user memory;
- project memory;
- procedural memory;
- task and workflow state;
- memory importance and confidence;
- memory consolidation and reflection;
- temporal knowledge-graph relationships;
- memory lifecycle, correction, and forgetting policies.

MemPalace gives us a strong retrieval foundation, but A.S.T.A. will own the
higher-level memory architecture.

## Development rule

Do not put MemPalace-specific imports into the kernel or UI. New memory
features should go through the A.S.T.A. memory abstraction first.
