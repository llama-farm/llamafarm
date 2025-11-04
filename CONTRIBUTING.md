# Contributing to LlamaFarm

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> If you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Share it on social media
> - Refer to this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues


## Table of Contents

- [Contributing to LlamaFarm](#contributing-to-llamafarm)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [I have a question (and I don't want to read all of this!!)](#i-have-a-question-and-i-dont-want-to-read-all-of-this)
  - [I want To contribute](#i-want-to-contribute)
    - [Reporting bugs](#reporting-bugs)
      - [Before submitting a bug report](#before-submitting-a-bug-report)
      - [How do I submit a good bug report?](#how-do-i-submit-a-good-bug-report)
    - [Suggesting enhancements](#suggesting-enhancements)
      - [Before submitting an enhancement](#before-submitting-an-enhancement)
      - [How do I submit a good enhancement suggestion?](#how-do-i-submit-a-good-enhancement-suggestion)
    - [Your first code contribution](#your-first-code-contribution)
  - [Styleguides](#styleguides)
    - [Code Formatting](#code-formatting)
      - [Pre-commit Hooks](#pre-commit-hooks)
      - [What Happens When Pre-commit Fixes Errors?](#what-happens-when-pre-commit-fixes-errors)
    - [Commit Messages](#commit-messages)
  - [Join the LlamaFarm project team](#join-the-llamafarm-project-team)


## Code of Conduct

This project and everyone participating in it is governed by the
[LlamaFarm Code of Conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to [oss@llamafarm.dev](mailto:oss@llamafarm.dev).


## I have a question (and I don't want to read all of this!!)

Feel free to start a thread within our [discussion forum](/discussions). We do suggest reading the docs and searching through existing [issues](/issues) and discussion threads first.





## I want To contribute

> ### Legal notice
> When contributing to this project, you must agree that you have authored 100% of the content (use of LLMs/AI is permitted), that you have the necessary rights to the content, and that the content you contribute may be provided under the project license.

### Reporting bugs


#### Before submitting a bug report

A good bug report shouldn't leave others needing to chase you down for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://docs.llamafarm.dev). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Ask your favorite AI tools to help confirm the bug.
- Collect information about the bug:
- Stack trace (Traceback)
- OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
- Version of the interpreter, compiler, SDK, runtime environment, package manager, model, etc depending on what seems relevant.
- Possibly your input and the output
- Can you reliably reproduce the issue? And can you also reproduce it with older versions?


#### How do I submit a good bug report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to [security@llamafarm.dev](mailto:security@llamafarm.dev).


We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](/issues/new).
- Explain the behavior you would expect vs. the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `bug`, as well as possibly other tags (such as `critical`), and the issue will be left to be [implemented by someone](#your-first-code-contribution).




### Suggesting enhancements

This section guides you through submitting an enhancement suggestion for LlamaFarm, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


#### Before submitting an enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://docs.llamafarm.dev) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.


#### How do I submit a good enhancement suggestion?

Enhancement suggestions are tracked as [GitHub issues](/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
- **Explain why this enhancement would be useful** to most LlamaFarm users. You may also want to point out the other projects that solved it better and which could serve as inspiration.



### Your first code contribution

Unsure where to begin contributing to LlamaFarm? You can start by looking through these beginner and help-wanted issues:

[Good first issues](https://github.com/search?utf8=%E2%9C%93&q=is:open+is:issue+label:%22good%20first%20issue%22+user:llama-farm+sort:comments-desc&type=issues) - issues which should only require a few lines of code, and a test or two.
[Help wanted issues](https://github.com/search?utf8=%E2%9C%93&q=is:open+is:issue+label:%22help%20wanted%22+user:llama-farm+sort:comments-desc&type=issues) - issues which should be a bit more involved than beginner issues.
Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.


## Styleguides

### Code Formatting

#### Pre-commit Hooks
This repository uses pre-commit hooks to automatically format code before commits. Ruff configuration is shared across all Python components via `ruff.toml` at the repository root.

**Installation:**
```bash
# From the repository root
uvx pre-commit install
```

The hooks will automatically run when you commit changes. To manually run the hooks on staged files:
```bash
uvx pre-commit run
```

To run on all files:
```bash
uvx pre-commit run --all-files
```

#### What Happens When Pre-commit Fixes Errors?

When the pre-commit hooks automatically fix formatting issues, the commit will be **blocked** and you'll need to re-stage the fixed files:

```bash
# 1. Try to commit
git add runtimes/universal/models/base.py
git commit -m "feat(runtime): add new model feature"

# Output: "Files were modified by this hook..."
# The commit is blocked because hooks fixed issues

# 2. Review the automatic fixes
git diff runtimes/universal/models/base.py

# 3. Re-stage the fixed files and commit again
git add runtimes/universal/models/base.py
git commit -m "feat(runtime): add new model feature"

# This time it succeeds! ✅
```

**Tip:** You can also run `uvx pre-commit run` before committing to catch and fix issues early.

#### Shared Ruff Configuration

All Python components use a shared ruff configuration located at `ruff.toml` in the repository root. This eliminates duplication and ensures consistency across all Python code.

Component-specific exclusions (like `datamodel.py` for config or `chroma_db/` for rag) are defined using `extend-exclude` in each component's `pyproject.toml`.

### Commit Messages
We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to keep things structured and make changelog management easy.

Please ensure commits follow the pattern: `type(component): description`.

Example: `fix(cli): cmd 'lf project list' doesn't honor cwd flag`

## Join the LlamaFarm project team
We welcome contributions and would love to add new maintainers over time!

If you'd like to join the team of maintainers, you should start by becoming an active participant and contributor to the project. Get involved by fixing/implementing issues. As we see a pattern of consistent engagement, adherence to the code of conduct, positive contributions, and so on, an existing maintainer may reach out to inquire about your interest in joining the team.
