# Contributing to Eval Harness

Thank you for considering contributing to Eval Harness! We appreciate your help.

## How to Contribute

### Reporting Issues
- Use the [issue tracker](https://github.com/onicarps/eval-harness/issues) to report bugs or request features
- Please check if the issue has already been reported before submitting a new one
- Include as much detail as possible: steps to reproduce, expected vs actual behavior, logs, etc.

### Pull Requests
1. Fork the repository
2. Create a new branch for your feature or bugfix (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for any new functionality
5. Ensure all tests pass (`pytest tests/ -v`)
6. Run the linter (`ruff check src tests` and `ruff format --check src tests`)
7. Run type checking (`mypy --config-file pyproject.toml src`)
8. Commit your changes (`git commit -m "feat: add amazing feature"`)
9. Push to your branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## Development Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/onicarps/eval-harness.git
cd eval-harness

# Install in development mode with development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test module
pytest tests/test_cli.py -v
```

### Code Quality
We use the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pytest** for testing

Run all checks locally before submitting:
```bash
ruff check src tests
ruff format --check src tests
mypy --config-file pyproject.toml src
pytest tests/ -v
```

### Making Changes
Please follow these guidelines when making changes:

#### Branching
- Use descriptive branch names: `feature/`, `bugfix/`, `docs/`, etc.
- Keep branches focused on a single topic

#### Commits
- Write clear, descriptive commit messages
- Use the format: `type: description` (e.g., `feat: add user authentication`)
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

#### Code Style
- Follow PEP 8 and PEP 257 (docstring conventions)
- Use type hints for all public functions
- Add Google-style docstrings to all public functions and classes
- Keep lines to 100 characters maximum
- Use 4 spaces for indentation (no tabs)

### Getting Help
If you need help, please:
1. Check the documentation in the `README.md`
2. Look at existing code for examples
3. Ask in the issue tracker

Thank you for contributing to Eval Harness!
