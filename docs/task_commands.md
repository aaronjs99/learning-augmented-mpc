# Task Commands (Repository Maintenance)

Use these commands for lightweight repo checks.

## Show current status
```bash
git status --short --branch
```

## Show current structure (excluding .git)
```bash
find . -path ./.git -prune -o -print | sort
```

## Validate markdown files are tracked
```bash
git ls-files "*.md"
```

## Preview recent commit history
```bash
git log --oneline --decorate --max-count=10
```
