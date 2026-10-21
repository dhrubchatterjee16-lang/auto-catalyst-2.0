# Contributing Guidelines

Thank you for your interest in contributing to this scientific open-source project! We welcome pull requests from researchers, engineers, and students in chemical engineering, physics-informed machine learning, and computational chemistry.

---

## 🛠️ Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Create and activate a clean virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```
3. Install dependencies in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   pip install pytest pytest-cov
   ```

---

## 🧪 Scientific & Software Standards

All contributions must adhere to our rigorous scientific software standards:
1. **Physical & Thermodynamic Consistency:**
   - Ensure conservation laws (mass, energy, momentum) are strictly preserved.
   - For thermodynamic surrogates, verify continuous derivatives ($C^2$) and positive definite hessians where convexity is required.
2. **Test Coverage:**
   - Every new function or model layer must be accompanied by a unit test in `tests/`.
   - Run the local test suite before opening a PR:
     ```bash
     pytest tests/ -v
     ```
3. **Commit Messages:**
   - We follow conventional commit conventions (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).

---

## 📬 Submitting a Pull Request

1. Create a descriptive feature branch (`git checkout -b feature/improved-fugacity`).
2. Commit your changes and push to your fork.
3. Open a Pull Request against `main`. Automated GitHub Actions CI will run and verify unit tests across Python environments.
