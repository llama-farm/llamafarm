# Contributing to LlamaFarm 🌾

First off, thank you for considering contributing to LlamaFarm! It's people like you that make LlamaFarm such a great tool for the AI community.

## 🌱 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples** to demonstrate the steps
* **Describe the behavior you observed** and what behavior you expected
* **Include screenshots** if possible
* **Include your configuration** (device, OS, Node.js version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a detailed description** of the suggested enhancement
* **Provide specific examples** to demonstrate the feature
* **Describe the current behavior** and explain what behavior you expected to see instead
* **Explain why this enhancement would be useful** to most LlamaFarm users

### Your First Code Contribution

Unsure where to begin? You can start by looking through these beginner and help-wanted issues:

* [Beginner issues](https://github.com/llamafarm/llamafarm-cli/labels/beginner) - issues which should only require a few lines of code
* [Help wanted issues](https://github.com/llamafarm/llamafarm-cli/labels/help%20wanted) - issues which should be a bit more involved

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## 🚜 Development Process

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/llamafarm-cli.git
cd llamafarm-cli

# Add upstream remote
git remote add upstream https://github.com/llamafarm/llamafarm-cli.git

# Install dependencies
npm install

# Run in development mode
npm run dev

# Run tests
npm test
```

### Code Style

We use TypeScript and follow these conventions:

* 2 spaces for indentation
* Use meaningful variable names
* Add comments for complex logic
* Follow the existing code style
* Keep the farming theme alive! 🌾

### Project Structure

```
llamafarm-cli/
├── src/
│   ├── commands/      # CLI commands (plant, harvest, etc.)
│   ├── templates/     # Code generation templates
│   ├── utils/         # Utility functions
│   └── cli.ts         # Main CLI entry point
├── tests/             # Test files
└── docs/              # Documentation
```

### Adding a New Command

1. Create a new file in `src/commands/`
2. Export an async function that handles the command
3. Add the command to `src/cli.ts`
4. Add tests in `tests/`
5. Update the README.md

Example:
```typescript
// src/commands/mycommand.ts
import chalk from 'chalk';

export async function mycommandCommand(options: any) {
  console.log(chalk.green('🌾 My new command!'));
  // Implementation
}
```

### Commit Messages

We follow conventional commits:

* `feat:` New feature
* `fix:` Bug fix
* `docs:` Documentation only changes
* `style:` Code style changes (formatting, etc)
* `refactor:` Code change that neither fixes a bug nor adds a feature
* `test:` Adding missing tests
* `chore:` Changes to the build process or auxiliary tools

Examples:
```
feat: add support for Qdrant vector database
fix: correct port assignment in harvest command
docs: update README with new greenhouse options
```

## 🌻 Community Guidelines

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Farm Values 🦙

* **Simplicity**: Make AI deployment accessible to everyone
* **Local-First**: Privacy and ownership matter
* **Open Source**: Share knowledge freely
* **Fun**: Keep the farming theme enjoyable!

## 📋 Issue Templates

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g. macOS 14.0]
 - Node version: [e.g. 18.17.0]
 - LlamaFarm version: [e.g. 0.1.0]
```

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other context or screenshots.
```

## 🎯 Areas We Need Help

* **Model Support**: Adding support for more models
* **Vector Databases**: Integrating Qdrant, Pinecone, Weaviate
* **Agent Frameworks**: Adding AutoGen, CrewAI templates
* **Documentation**: Tutorials, recipes, and guides
* **Testing**: Increasing test coverage
* **Performance**: Optimizing compilation and runtime
* **Platforms**: Windows ARM, mobile support

## 💬 Questions?

Feel free to:
* Open an issue with the question label
* Join our Discord community
* Email us at contribute@llamafarm.ai

Thank you for helping make LlamaFarm better! 🌾🦙