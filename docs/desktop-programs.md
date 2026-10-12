# Writing and opening programs on the desktop

"Write a Snake game in Python and run it" needs a screen and a keyboard, which
the code sandbox doesn't have. For programs you want to see or use (games,
windows, interactive apps) ASTA plans an `open_program` step instead of a
sandbox run:

1. It writes the file (with the Python standard library, tkinter for games,
   unless you name a package) and verifies it on disk.
2. It asks first: *"Can I open snake.py on your desktop? It runs outside the
   sandbox."* Nothing runs until you approve.
3. It starts `python snake.py` from the workspace (the project's own `.venv`
   if it has one, else `ASTA_PROJECT_PYTHON`, else `python` on PATH) and
   watches the first few seconds (`ASTA_PROGRAM_SETTLE_SECONDS`, default 3).
4. If the program crashes right away, the error (for example
   `No module named 'pygame'`) goes back to the planner, which fixes the code
   and asks to open it again.

Only `.py` files inside the active workspace can be opened. Tests and scripts
that print a result and exit still run in the sandbox.

Larger files need room in the coding model's reply: the defaults are now
`ASTA_CODING_LLM_CONTEXT_SIZE=8192` and `ASTA_CODING_AGENT_MAX_OUTPUT_TOKENS=4096`,
and the model writes each file's content only once.

While a project task runs, everyday commands that don't touch the project
(opening apps, searching, playing music) run right away beside it.
