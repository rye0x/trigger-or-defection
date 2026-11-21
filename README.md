# TARA Project

Working repository for the TARA project: notes, logs, and supporting material.

## Contents

| Path | Purpose |
|------|---------|
| `project_log.md` | Dated research log: what was done, expected vs. actual results, next steps |
| `paper/` | The manuscript — **a separate repo**: [tara-paper](https://github.com/rye0x/tara-paper) (ignored here) |

## Setup

```bash
git clone https://github.com/rye0x/tara_project.git
cd tara_project
git clone https://github.com/rye0x/tara-paper.git paper   # optional: the paper
```

## Working with the two repos

`paper/` is listed in `.gitignore`, so each folder commits to its own repo:

```bash
# project changes (from tara_project/)
git add -A && git commit -m "..." && git push

# paper changes
cd paper && git add -A && git commit -m "..." && git push
```
