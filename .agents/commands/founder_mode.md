you're working on an experimental feature that didn't get the proper ticketing and pr stuff set up.

assuming you just made a commit, here are the next steps:

get the sha of the commit you just made (if you didn't make one, read .agents/commands/commit.md and make one)

read .agents/commands/github_issues.md - think deeply about what you just implemented, then create a github issue about what you just did - it should have ### headers for "problem to solve" and "proposed solution"

fetch the issue to get the recommended git branch name

git checkout main

git checkout -b 'BRANCHNAME'

git cherry-pick 'COMMITHASH'

git push -u origin 'BRANCHNAME'

gh pr create --fill

read '.agents/commands/describe_pr.md' and follow the instructions