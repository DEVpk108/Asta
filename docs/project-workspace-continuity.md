# Project/workspace continuity acceptance test

## Goal

After an A.S.T.A. restart, workspace-specific follow-up requests should refer to
the same existing project root and the recent files touched through the
workspace tools—unless the user explicitly selects another root.

## Acceptance criteria

1. On a successful workspace-file read or write, the relative path is moved to
   the front of the recent-file list (maximum 20 entries).
2. On shutdown/restart, the project name, absolute root, sanitized repository
   URL, branch, and safe relative active/recent file paths are restored when
   the saved project directory still exists.
3. `ASTA_WORKSPACE_PATH` overrides the saved project. Switching roots clears
   the previous project's active/recent paths before publishing the new
   workspace state.
4. If the saved root no longer exists or the state file is invalid, A.S.T.A.
   falls back to the configured path, current Git root, or launch directory.
5. The JSON state is local, versioned, atomically replaced, and excludes file
   contents, environment variables, hardware details, and credentials. Absolute
   or sensitive file paths are not stored as recent/active files.
6. An unwritable or corrupt continuity file must not prevent startup or cause a
   successful workspace file operation to fail.
7. A persisted project path and recent file names are context hints, not
   proof that a file still exists or is unchanged; tools must re-read content
   before acting on it.

## Explicitly out of scope

This slice does not persist task plans, tool approvals, task evidence, or
in-progress task state. It must not automatically replay a previous command or
resume a task after restart. Task checkpointing requires a separate, explicit
permission and recovery design.