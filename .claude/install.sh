#!/bin/bash
# install.sh — link this repo's Claude Code skills + memory into your user paths.
#
# Run this once on each new machine/cluster after cloning the repo. It creates
# symlinks so edits flow back to the repo and `git pull` updates your skills
# and memory in one step.
#
# Usage:
#   ./install.sh                          # auto-detect primary workdir from $PWD
#   ./install.sh /data/home/$USER         # explicit primary workdir
#
# The primary workdir is the path Claude Code shows as "Primary working
# directory" in /context. Memory files are scoped to this path via the
# encoded directory name under ~/.claude/projects/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_DIR="$REPO_DIR/.claude"

WORK_DIR="${1:-$(pwd)}"
encoded=$(echo "$WORK_DIR" | sed 's|/|-|g')   # /data/home/x -> -data-home-x

echo "Repo:           $REPO_DIR"
echo "Primary workdir: $WORK_DIR"
echo "Memory target:  ~/.claude/projects/$encoded/memory"
echo

link_file() {
    local src="$1" dst="$2"
    if [[ -L "$dst" ]]; then
        echo "skip (already symlink): $dst"
    elif [[ -e "$dst" ]]; then
        echo "BACKUP existing real file: $dst -> $dst.bak"
        mv "$dst" "$dst.bak"
        ln -s "$src" "$dst"
        echo "linked: $dst -> $src"
    else
        ln -s "$src" "$dst"
        echo "linked: $dst -> $src"
    fi
}

# 1. Skills — global, not path-encoded
mkdir -p "$HOME/.claude/skills"
for skill_dir in "$CLAUDE_DIR/skills/"*/; do
    name=$(basename "$skill_dir")
    link_file "$skill_dir" "$HOME/.claude/skills/$name"
done
echo

# 2. Memory — path-encoded per primary workdir
mem_dir="$HOME/.claude/projects/$encoded/memory"
mkdir -p "$mem_dir"
for mem_file in "$CLAUDE_DIR/memory/"*.md; do
    name=$(basename "$mem_file")
    link_file "$mem_file" "$mem_dir/$name"
done

echo
echo "Done. Restart Claude Code (or start a new session) to pick up changes."
