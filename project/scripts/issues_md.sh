#!/usr/bin/env bash
# Write ISSUES.md: all GitHub issues grouped by roadmap week, with board
# status and assignees. Rerun to refresh. Needs gh with the `project` scope.
set -euo pipefail

ORG=DeepLearningTukeDreamerV3
REPO=$ORG/dreamerv3-crafter
PROJECT=1
OUT=${1:-ISSUES.md}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

gh issue list -R "$REPO" --state all --limit 500 \
  --json number,title,state,assignees,labels,milestone,body > "$tmp/issues.json"
gh project item-list "$PROJECT" --owner "$ORG" --limit 500 --format json \
  --jq '[.items[] | select(.content.number) | {key: (.content.number|tostring), value: .status}] | from_entries' \
  > "$tmp/status.json"

jq -r --slurpfile st "$tmp/status.json" --arg now "$(date '+%Y-%m-%d %H:%M')" --arg repo "$REPO" '
  def week: (.body | capture("Roadmap week (?<w>[0-9]+)").w // "?") ;
  def status: if .state == "CLOSED" then "Done" else ($st[0][.number|tostring] // "Todo") end ;
  def box: if status == "Done" then "[x]" else "[ ]" end ;
  def who: if (.assignees|length) == 0 then "_unassigned_" else ([.assignees[].login] | map("@" + .) | join(", ")) end ;
  "# Issues\n",
  "Generated \($now) by `project/scripts/issues_md.sh`. Do not edit; rerun to refresh.\n",
  "Board: https://github.com/orgs/DeepLearningTukeDreamerV3/projects/1\n",
  (group_by(week | tonumber? // 99)[] |
    "## Week \(.[0] | week)\n",
    (sort_by(.number)[] |
      "- \(box) [#\(.number)](https://github.com/\($repo)/issues/\(.number)) \(.title) — **\(status)** · \(who) · \(.milestone.title // "no milestone") · \([.labels[].name] | join(", "))"),
    "")
' "$tmp/issues.json" > "$OUT"

echo "Wrote $OUT"
