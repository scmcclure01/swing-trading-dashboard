---
name: github-push
description: Push updated files from the workspace folder to the GitHub repo. Use this skill whenever the user says "push to github", "deploy to github", "update github", "push the changes", or wants to sync the Streamlit dashboard. This is the standard workflow for deploying app.py and related files.
---

# GitHub Push

Repo: https://github.com/scmcclure01/swing-trading-dashboard (public)
Git user: Scot McClure / scmcclure01@gmail.com

**IMPORTANT:** Google Drive blocks git writes — never init .git in the Drive mount. Always clone to /tmp first.

The bash session mount path for the workspace is: `/sessions/*/mnt/Swing Trading Framework/`
Run `ls /sessions/` first to get the actual session name.

---

## Step 1 — Get the PAT

Ask Scot: "Please provide your GitHub PAT to push."

Do not store the token. Use it only within this session.

---

## Step 2 — Identify what changed

Tell Scot what files you're about to push (e.g., `app.py`, any other modified files). Confirm before pushing.

---

## Step 3 — Clone, sync, push

```bash
# Get actual session name
SESSION=$(ls /sessions/ | head -1)
WORKSPACE="/sessions/${SESSION}/mnt/Swing Trading Framework"

# Clone repo to temp location
git clone https://<TOKEN>@github.com/scmcclure01/swing-trading-dashboard.git /tmp/dashboard_push

# Configure git identity
cd /tmp/dashboard_push
git config user.email "scmcclure01@gmail.com"
git config user.name "Scot McClure"

# Sync files from Drive workspace into repo
# Copy specific changed files — do NOT bulk rsync everything (avoid overwriting repo-only files)
cp "${WORKSPACE}/app.py" /tmp/dashboard_push/app.py
# Add other changed files here as needed

# Commit and push
git add -A
git commit -m "<commit message describing what changed>"
git push origin main
```

---

## Step 4 — Confirm

Report the commit hash and confirm the push succeeded. Streamlit Cloud auto-deploys on push — no further action needed.

---

## Notes

- The Drive mount path changes each session (`/sessions/<session-name>/mnt/`) — always resolve it dynamically via `ls /sessions/`
- Only copy files that were actually changed — don't blindly rsync the entire workspace
- If push fails with auth error, the PAT may be expired — ask Scot to generate a new one at github.com → Settings → Developer settings → Personal access tokens
