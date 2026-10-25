"""Small auditable runner for this plain-Python notebook; no Jupyter dependency.

Not a general IPython kernel: magics, widgets, async cells and shell escapes are
unsupported. Every shipped code cell uses normal Python and is run in order.
"""
import ast
import base64
import contextlib
import io
import json
import os
import traceback
from pathlib import Path

_SINK = None


def display_table(frame):
    if _SINK is not None:
        _SINK({"output_type": "display_data", "metadata": {},
               "data": {"text/plain": frame.to_string(index=False), "text/html": frame.to_html(index=False)}})
    else:
        try:
            from IPython.display import display
            display(frame)
        except ImportError:
            print(frame.to_string(index=False))


def display_png(path):
    if _SINK is not None:
        _SINK({"output_type": "display_data", "metadata": {}, "data": {
            "image/png": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
            "text/plain": "Continuous reference overlays from supplied detector data"}})
    else:
        try:
            from IPython.display import Image, display
            display(Image(filename=str(path)))
        except ImportError:
            print(f"Figure: {path}")


def execute_notebook(source, destination, cwd):
    global _SINK
    notebook = json.loads(Path(source).read_text())
    namespace = {"__name__": "__main__"}
    old_cwd = Path.cwd()
    destination = Path(destination).resolve()
    count = 0
    failure = None
    try:
        os.chdir(cwd)
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            cell["execution_count"] = count
            cell["outputs"] = []
            _SINK = cell["outputs"].append
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                code = "".join(cell["source"])
                # Compile validates ordinary Python syntax before executing.
                ast.parse(code)
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(code, f"<notebook cell {i}>", "exec"), namespace)
            except Exception as exc:
                failure = exc
                cell["outputs"].append({"output_type": "error", "ename": type(exc).__name__,
                                        "evalue": str(exc), "traceback": traceback.format_exc().splitlines()})
            for name, buffer in (("stdout", stdout), ("stderr", stderr)):
                if buffer.getvalue():
                    cell["outputs"].append({"output_type": "stream", "name": name, "text": buffer.getvalue()})
            if failure:
                break
    finally:
        _SINK = None
        os.chdir(old_cwd)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    if failure:
        raise failure
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()
    from saa_baseline.notebook_runner import execute_notebook as run
    print(f"Executed {run(args.source, args.destination, args.cwd)} code cells.")
