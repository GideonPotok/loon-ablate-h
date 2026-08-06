## Session Context

You are running in a standalone Stork session (stork-wrapper).
Unlike the full harness, this is a per-spawn session — no daemon, no channels.

- **Session name:** 20260806212247-3mcf
- **Session directory:** /Users/gideonpotok/repos/loon-ablate-h/
- **Current date (UTC):** 2026-08-06

Session directory layout:
- `MEMORY.md` — persistent notes (survives across sessions for this directory)
- `scratch/` — your working directory for intermediate files
- `.stork/sessions/` — resume files for conversation continuity
- `.stork/skills/` — project-local skills (override user skills)
- `attachments/` — files available for this session

Write to `MEMORY.md` in the session directory to persist notes across sessions.
The `STORK_USER_CWD` env var points to the session directory (`/Users/gideonpotok/repos/loon-ablate-h`).
The `STORK_WORKSPACE` env var also points to the session directory.

To resume this session later, run:
  stork-wrapper --session 20260806212247-3mcf --session-dir /Users/gideonpotok/repos/loon-ablate-h [stork opts...]
