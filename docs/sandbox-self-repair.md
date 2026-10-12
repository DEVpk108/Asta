# Sandbox self-repair

Project commands run inside a Docker/Podman sandbox. When a run fails because
the sandbox itself isn't ready (not because of the user's code), ASTA repairs
the environment and continues the task instead of stopping.

## Flow

1. **Detect** – `system.run_command` fails with a sandbox setup error
   (`Sandbox unavailable:` / `Sandbox container creation failed.`).
   Replanning the code can't fix that, so the task pauses for repair.
2. **Observe** – `core/autonomy/sandbox_setup.py` checks, in order:
   is the `docker`/`podman` CLI installed? does `docker info` answer?
   does `docker image inspect asta-python-sandbox:1` succeed?
3. **Fix** – only fixed, vetted recipes:
   - engine stopped → start Docker Desktop (Windows/macOS/Linux) or
     `podman machine start`, then poll `docker info` (default 180 s);
   - image missing → `docker build --pull -t asta-python-sandbox:1 -f sandbox/Dockerfile <ASTA>`
     from ASTA's own Dockerfile (default limit 20 min);
   - CLI missing → explain; ASTA never installs software.
4. **Verify** – the engine and image checks must pass after the fix.
5. **Resume** – the failed step is rerun. A command the user already approved
   reruns once without asking again (same tool, arguments, task and step,
   within 15 minutes). Repair is attempted once per task.

ASTA says what it observed and did, and records it as task evidence
(`capability_setup` with `observed` and `actions`).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ASTA_SANDBOX_SELF_REPAIR` | `1` | `0` restores the old "stop and explain" behaviour |
| `ASTA_SANDBOX_AUTO_BUILD` | `1` | `0` never builds the image; ASTA tells you the command |
| `ASTA_SANDBOX_STARTUP_TIMEOUT` | `180` | seconds to wait for the engine after starting it |
| `ASTA_SANDBOX_BUILD_TIMEOUT` | `1200` | seconds allowed for the image build |
| `ASTA_DOCKER_DESKTOP_PATH` | auto | path to `Docker Desktop.exe` if it's installed somewhere unusual |

Tests set `ASTA_SANDBOX_SELF_REPAIR=0` (tests/conftest.py) so a test run never
starts Docker or builds images on your machine.
