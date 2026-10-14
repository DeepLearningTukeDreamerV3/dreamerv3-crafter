# Git workflow: how to push an update

`main` is protected. You **cannot** push to it directly. Every change goes
through a branch and a pull request (PR) that a teammate approves. GitHub
then squashes the PR into one commit on `main`.

If something goes wrong, jump to [Rescue](#rescue). Do not use
`git push --force` on anything unless this page tells you to.

## One-time setup

1. Accept the invitation to the GitHub org
   [DeepLearningTukeDreamerV3](https://github.com/DeepLearningTukeDreamerV3).
2. Install git and the GitHub CLI (`gh`), then log in:
   ```sh
   gh auth login          # GitHub.com -> HTTPS -> login with a web browser
   gh auth setup-git      # git uses your gh login for push/pull
   ```
3. Clone and set your name:
   ```sh
   git clone https://github.com/DeepLearningTukeDreamerV3/dreamerv3-crafter.git
   cd dreamerv3-crafter
   git config user.name  "Your Name"
   git config user.email "you@example.com"   # an email on your GitHub account
   ```

## The loop (every change)

### 1. Start from an up-to-date `main`

```sh
git switch main
git pull
```

### 2. Create a branch

Name: `<type>/<short-name>`. Types: `infra`, `exp`, `analysis`, `docs`, `fix`.

```sh
git switch -c docs/architecture-shapes
```

### 3. Work, then check what changed

```sh
git status             # which files changed
git diff               # what changed inside them
```

Never commit checkpoints, replay buffers, `.sif` images, `.env` or big
files. `git status` should only show files you meant to change.

### 4. Commit

```sh
git add project/docs/architecture.md       # add files by name, not "git add ."
git commit -m "Add tensor shapes to architecture doc"
```

Commit message: short, imperative ("Add ...", "Fix ...", "Update ...").
Many small commits are fine; they are squashed at merge.

### 5. Push the branch

```sh
git push -u origin docs/architecture-shapes   # first push of this branch
git push                                      # later pushes
```

### 6. Open a pull request

```sh
gh pr create --fill          # or click "Compare & pull request" on GitHub
```

Fill the checklist in the PR description. Link the issue with
`Closes #12` so it closes automatically.

### 7. Review

- A teammate is asked for review automatically. Ping them if needed.
- To change something after review: edit, commit, `git push` again. The PR
  updates itself. The reviewer has to approve again.
- Answer or resolve every comment. Unresolved comments block the merge.

### 8. Merge

When approved and the checks are green, click **Squash and merge** on the
PR page. The branch is deleted on GitHub automatically.

### 9. Clean up locally

```sh
git switch main
git pull
git branch -d docs/architecture-shapes
```

Then start again at step 2 for the next change.

## `main` moved while I was working

If the PR says "This branch is out-of-date", click **Update branch** on the
PR page. Then, locally:

```sh
git pull
```

Doing it in the terminal instead:

```sh
git switch docs/architecture-shapes
git fetch origin
git merge origin/main
git push
```

## Merge conflicts

`git merge` or **Update branch** may stop with `CONFLICT`. Then:

1. `git status` lists the conflicted files.
2. Open each file. Find the blocks between `<<<<<<<`, `=======` and
   `>>>>>>>`. Keep the right content and delete the markers.
3. Finish:
   ```sh
   git add <file>
   git commit              # accept the default message
   git push
   ```

Not sure which side is right? Run `git merge --abort` to cancel and ask the
author of the other change.

## Rescue

**`git push` to `main` was rejected.** Expected; `main` is protected. Move
your commits to a branch:
```sh
git switch -c fix/my-change        # new branch keeps your commits
git push -u origin fix/my-change
git switch main
git reset --hard origin/main       # local main back to GitHub's version
```
Then open a PR (step 6).

**I committed on `main` but have not pushed.** Same commands as above.

**Undo my last commit but keep the changes.**
```sh
git reset --soft HEAD~1
```

**I committed a big or secret file.** If it is not pushed yet:
```sh
git reset --soft HEAD~1
git restore --staged <file>
```
Add the path to `.gitignore`. If it is already pushed, tell the team. If it
was a password or token, revoke it now; deleting the commit does not
un-leak it.

**Throw away my local edits to a file.**
```sh
git restore <file>                 # cannot be undone
```

**I am lost.** Run `git status` and `git log --oneline -5`, paste the
output in the team chat. Do not try random commands with `--force`.

## Rules enforced by GitHub on `main`

- No direct push, no force push, no deletion.
- PR needs 1 approving review. New commits drop old approvals.
- All review conversations resolved.
- `Large file check` passes (no file over 5 MB).
- Squash merge only. Branches are deleted after merge.
