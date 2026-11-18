#!/usr/bin/env bash
# Rebuild the integration branch (default: local-all) from upstream master plus the
# feature branches listed in a file, and push it to the fork.
#
#   sync_local_all.sh [branches.txt]
#
# Environment (all optional):
#   UPSTREAM_URL     default https://github.com/pymeasure/pymeasure.git
#   UPSTREAM_BRANCH  default master
#   REMOTE           the fork's remote name, default origin
#   TARGET           integration branch name, default local-all
#   FORCE=1          rebuild even if no source branch changed
#   DRY_RUN=1        build but do not push
#   UNION_FILES      space-separated files merged with git's 'union' strategy (both sides' added
#                    lines are kept, so several branches can each add a changelog entry at the same
#                    spot without conflicting). Default CHANGES.rst; set to empty to disable.
#
# Exit status: 0 = up to date or built with every branch merged, 1 = built but at least one
# branch was skipped (merge conflict) or a listed branch is missing, 2 = setup error.
set -uo pipefail

LIST=${1:-branches.txt}
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/pymeasure/pymeasure.git}
UPSTREAM_BRANCH=${UPSTREAM_BRANCH:-master}
REMOTE=${REMOTE:-origin}
TARGET=${TARGET:-local-all}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
UNION_FILES=${UNION_FILES-CHANGES.rst}

warn() { echo "::warning::$*"; }
die() { echo "::error::$*"; exit 2; }

[ -f "$LIST" ] || die "branch list '$LIST' not found"
git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"

git remote get-url upstream >/dev/null 2>&1 || git remote add upstream "$UPSTREAM_URL"
git fetch --quiet --no-tags upstream \
    "+refs/heads/$UPSTREAM_BRANCH:refs/remotes/upstream/$UPSTREAM_BRANCH" || die "fetch upstream failed"
git fetch --quiet --no-tags --prune "$REMOTE" \
    "+refs/heads/*:refs/remotes/$REMOTE/*" || die "fetch $REMOTE failed"
base=$(git rev-parse "refs/remotes/upstream/$UPSTREAM_BRANCH")

# ---- expand the branch list (comments, blank lines and glob patterns allowed) -------------------
wanted=()
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue
    if [[ $line == *[\*\?\[]* ]]; then
        while IFS= read -r b; do wanted+=("$b"); done < <(
            git for-each-ref --format='%(refname:lstrip=3)' "refs/remotes/$REMOTE/$line" | sort)
    else
        wanted+=("$line")
    fi
done < "$LIST"

features=(); missing=()
for b in "${wanted[@]}"; do
    case "$b" in "$TARGET"|HEAD|"$UPSTREAM_BRANCH") continue ;; esac
    [[ " ${features[*]} " == *" $b "* ]] && continue
    if git rev-parse --verify -q "refs/remotes/$REMOTE/$b" >/dev/null; then
        features+=("$b")
    else
        missing+=("$b")
    fi
done
for b in "${missing[@]}"; do warn "listed branch '$b' does not exist on $REMOTE"; done

# ---- skip if nothing changed since the last build ----------------------------------------------
sources="upstream/$UPSTREAM_BRANCH=$base"$'\n'
for b in "${features[@]}"; do sources+="$b=$(git rev-parse "refs/remotes/$REMOTE/$b")"$'\n'; done
hash=$(printf '%s' "$sources" | sha256sum | cut -c1-16)

old=$(git rev-parse -q --verify "refs/remotes/$REMOTE/$TARGET" || true)
if [ -n "$old" ] && [ "$FORCE" != 1 ]; then
    last=$(git log -1 --format=%B "$old" | sed -n 's/^Sources-hash: //p')
    if [ "$last" = "$hash" ]; then
        echo "$TARGET is up to date (sources $hash)"
        git log -1 --format=%B "$old" | grep -q '^Skipped' && \
            warn "the last $TARGET build skipped branches with conflicts; see its commit message"
        exit 0
    fi
fi

# ---- build in a separate worktree so this script is never overwritten --------------------------
root=$(git rev-parse --show-toplevel)
wt=$(mktemp -d)
trap 'cd /; git -C "$root" worktree remove --force "$wt" 2>/dev/null; git -C "$root" worktree prune' EXIT
git worktree add -q --detach "$wt" "$base" || die "worktree failed"
cd "$wt" || exit 2
git checkout -q -B "$TARGET" "$base"

# Per-invocation attributes file: never touches the repository's own attributes or config.
attrs=$(mktemp)
for f in $UNION_FILES; do echo "$f merge=union" >> "$attrs"; done
trap 'cd /; git -C "$root" worktree remove --force "$wt" 2>/dev/null; git -C "$root" worktree prune; rm -f "$attrs"' EXIT

merged=(); skipped=()
for b in "${features[@]}"; do
    sha=$(git rev-parse "refs/remotes/$REMOTE/$b")
    if git -c core.attributesFile="$attrs" merge --no-ff --no-edit -q -m "$TARGET: merge $b" "$sha" >/dev/null 2>&1; then
        merged+=("$b=${sha:0:10}")
    else
        git merge --abort 2>/dev/null || git reset --hard -q HEAD
        skipped+=("$b=${sha:0:10}")
        warn "merge conflict: '$b' was skipped"
    fi
done

msgfile=$(git rev-parse --git-dir)/build-msg
{
    echo "Rebuild $TARGET"
    echo
    echo "Upstream: ${UPSTREAM_BRANCH}=${base:0:10}"
    for m in "${merged[@]}"; do echo "Merged: $m"; done
    for s in "${skipped[@]}"; do echo "Skipped (conflict): $s"; done
    for m in "${missing[@]}"; do echo "Skipped (missing): $m"; done
    echo
    echo "Sources-hash: $hash"
} > "$msgfile"
git commit -q --allow-empty -F "$msgfile"
new=$(git rev-parse HEAD)
echo "built $TARGET -> ${new:0:10}: ${#merged[@]} merged, ${#skipped[@]} conflicted, ${#missing[@]} missing"

if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: not pushing"
else
    git push --force-with-lease="$TARGET:$old" "$REMOTE" "$TARGET" || die "push failed"
fi

[ ${#skipped[@]} -eq 0 ] && [ ${#missing[@]} -eq 0 ] || exit 1

