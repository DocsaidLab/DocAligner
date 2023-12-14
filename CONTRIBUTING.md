# Contributing to Docsaid

First off, thank you for considering contributing to Docsaid. It's people like you that make Docsaid such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our Issues to see if someone else in the community has already created a ticket. If not, go ahead and **make one**!

## Fork & create a branch

If this is something you think you can fix, then fork the repository and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```bash
git checkout -b feature/325-add-japanese-localization
```

## Get the test suite running

Make sure you're able to run the test suite. If you're having trouble getting it running, please create an issue and we'll help get you started.

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.

## Make a Pull Request

At this point, you should switch back to your main branch and make sure it's up to date with Docsaid's main branch:

```bash
git remote add upstream git@github.com:original/docsaidlab.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of main and push it!

```bash
git checkout feature/325-add-japanese-localization
git rebase main
git push --set-upstream origin feature/325-add-japanese-localization
```

Go to the Docsaid repository and you should see recently pushed branches.

Choose your branch and create a new Pull Request. Once you've created the Pull Request, maintainers will review your changes.

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

## Merging a PR (maintainers only)

A PR can only be merged into main by a maintainer if:

- It is passing CI.
- It has been approved by at least two maintainers. If it was a maintainer who opened the PR, only one extra approval is needed.
- It has no requested changes.
- It is up to date with current main.

Any maintainer is allowed to merge a PR if all of these conditions are met.

## Thank you for your contributions!

Your time and effort are greatly appreciated. Thank you for contributing to Docsaid!
