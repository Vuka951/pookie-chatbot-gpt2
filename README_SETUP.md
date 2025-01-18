## Setting Up a Virtual Environment

To keep the project dependencies isolated and avoid polluting the global Python environment, you can set up a virtual environment as follows:

### 1. Create a Virtual Environment

Run the following command in the project directory:

```
python3 -m venv .venv
```

This creates a `.venv` directory containing the virtual environment files.

---

### 2. Activate the Virtual Environment

Activate the virtual environment:

- **On macOS/Linux:**

  ```
  source .venv/bin/activate
  ```

- **On Windows (Command Prompt):**

  ```
  .venv\Scripts\activate
  ```

- **On Windows (PowerShell):**
  ```
  .\.venv\Scripts\Activate.ps1
  ```

You’ll see `(.venv)` in your terminal prompt, indicating the virtual environment is active.

---

### 3. Install Dependencies

Install project dependencies using `pip`:

```
pip install package_name
```

To save installed dependencies to a `requirements.txt` file:

```
pip freeze > requirements.txt
```

To install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

---

### 4. Deactivate the Virtual Environment

When done, deactivate the virtual environment:

```
deactivate
```

---
