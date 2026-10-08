# Sandboxed project commands

`system.run_command` is the project-task verification runner. It accepts only
Python and pytest, and it never executes the requested program on the host.
Every invocation requires the existing critical-risk confirmation.

## One-time setup

Install and start Docker Desktop (Windows) or Docker Engine/Podman (Linux), then
build the local image from the repository root. Pulling the base image happens
only during this explicit setup step; ASTA does not pull images while a task is
running.

```powershell
docker build --pull -t asta-python-sandbox:1 -f sandbox/Dockerfile .
```

Set these optional environment variables before launching ASTA if needed:

```powershell
$env:ASTA_SANDBOX_IMAGE = "asta-python-sandbox:1"
$env:ASTA_CONTAINER_RUNTIME = "docker" # or "podman"
```

The defaults match the image built above and Docker. If the runtime or image is
unavailable, the tool fails closed; it never falls back to host execution.

## Isolation contract

- The active workspace is copied to a temporary snapshot. Common secret names,
  credential directories, `.git`, virtual environments, dependency folders,
  symlinks, and non-regular files are omitted.
- The snapshot is mounted read-only at `/workspace`; no changes made by the
  command are written back to the project.
- The container has no network, a read-only root filesystem, no Linux
  capabilities, `no-new-privileges`, a non-root UID, process/memory/CPU limits,
  and a small temporary filesystem.
- Only the local prebuilt image is used (`--pull=never`). Standard output and
  standard error are bounded. Containers are removed after completion and
  force-removed after a timeout.
- The default limits are 20,000 directories, 10,000 files, 256 MiB total
  snapshot data, 16 MiB per file, 300 seconds, 128 KiB each for stdout/stderr,
  1 GiB memory/swap, and 64 processes.

The initial image provides Python and pytest, not ASTA's complete optional
runtime dependency set. To use another local image, build it intentionally and
set `ASTA_SANDBOX_IMAGE`; keep it Python-compatible and trusted. Runtime pulls
remain disabled. This meaningfully reduces host access, but containers are not
an absolute security guarantee against container-runtime or kernel
vulnerabilities; keep Docker/Podman and the chosen base image current.

Because execution uses a sanitized read-only snapshot, use workspace file
tools to make changes. Build tools that need to write should direct temporary
artifacts to `/tmp`; installing packages during task execution is intentionally
unsupported because networking is disabled.

## HUD status

The HUD shows the visible task stage, approval state, and whether Python/pytest
is running against a read-only snapshot with network access disabled. It uses
an indeterminate progress indicator because the current task contract does not
provide trustworthy completion percentages. It shows concise status and
verification evidence, not hidden chain-of-thought.

## Manual smoke check

With Docker running and the image built, choose a project containing a simple
`main.py`, approve the critical tool request, and ask ASTA to run it. Expected
tool evidence includes the return code, bounded output, `network: none`, and
`workspace_snapshot: read-only`. For the automated unit tests, the runtime is
faked so tests do not require Docker:

```powershell
.venv\Scripts\python -m pytest -q tests/test_sandbox_runner.py tests/test_workspace_project_task.py
```
