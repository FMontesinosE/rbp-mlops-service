# Contributing to RBP MLOps

Thank you for your interest in contributing to this project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rbp-mlops.git
cd rbp-mlops
```

2. Install dependencies:
```bash
make install
```

3. Run tests to verify setup:
```bash
make test
```

## Code Style

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

Run all checks before submitting:
```bash
make lint
make typecheck
make test
```

## Project Rules

Follow the guidelines in `CLAUDE.md`:
- Python 3.10+ with strict type hints
- Google-style docstrings for public functions only
- Real data in tests (no mock tensors in `src/`)
- Ruff formatting with 100-character line length

## Testing

Add tests for new features:
- Unit tests in `tests/test_*.py`
- Use fixtures from `tests/conftest.py`
- Aim for >80% coverage
- Include docstrings explaining what is tested

Example test structure:
```python
def test_feature_name() -> None:
    """Test description."""
    # Setup
    input_data = ...

    # Execute
    result = function_under_test(input_data)

    # Verify
    assert result == expected_value
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Brief description of changes"
```

3. Ensure all tests pass:
```bash
make lint && make typecheck && make test
```

4. Push to your fork and create a pull request

5. Address review feedback if needed

## Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Add details in body if needed

Good examples:
```
Add drift detection for 3-mer frequencies

Implement JS divergence calculation comparing incoming
sequences against training baseline distribution.
```

```
Fix boundary handling in sequence extraction

Prevent crashes when peaks are near chromosome edges
by adding coordinate validation and padding.
```

## Areas for Contribution

Potential areas to contribute:
- GPU support and mixed precision training
- Additional model architectures
- More comprehensive API tests
- Kubernetes deployment manifests
- Model explainability features
- Performance optimizations
- Documentation improvements

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation clarifications
- General questions

Please search existing issues first to avoid duplicates.

## Code Review

All submissions require review. We review for:
- Code quality and style compliance
- Test coverage
- Documentation clarity
- Performance implications
- Security considerations

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
