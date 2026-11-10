#!/usr/bin/env python
"""Execute one notebook top-to-bottom and record EVERYTHING next to it.

Usage:  python scripts/run_notebook.py notebooks/07-mlp-backprop/07_mlp_backprop.ipynb [--timeout 1800]

Produces, in the notebook's own folder:
    <name>.ipynb   executed notebook with all outputs embedded (overwritten in place)
    <name>.html    standalone rendered copy (figures included)
    run.log        per-cell timing + every stdout/stderr/traceback emitted by the cells
    figures/*.png  figures saved by the notebook via show()
    metrics.json   machine-readable summary written by the notebook

Exit code is 1 if any cell failed, but the partially executed notebook and the log are
still written, so the CI job can commit them for debugging.
"""
import argparse, os, sys, time, traceback, logging
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    a = ap.parse_args()

    nb_path = Path(a.notebook).resolve()
    folder = nb_path.parent
    (folder / "figures").mkdir(exist_ok=True)
    log_path = folder / "run.log"

    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")):
        h.setFormatter(fmt)
        logger.addHandler(h)

    logger.info("=== running %s ===", nb_path.name)
    logger.info("python %s | cwd for kernel: %s", sys.version.split()[0], folder)
    nb = nbformat.read(nb_path, as_version=4)
    total = len(nb.cells)
    t_start = time.perf_counter()
    state = {"t": 0.0}

    def on_start(cell, cell_index, **kw):
        if cell.cell_type == "code":
            state["t"] = time.perf_counter()
            first = cell.source.strip().splitlines()[0][:90] if cell.source.strip() else ""
            logger.info("[cell %02d/%02d] START  %s", cell_index + 1, total, first)

    def on_done(cell, cell_index, execute_reply, **kw):
        if cell.cell_type != "code":
            return
        dt = time.perf_counter() - state["t"]
        logger.info("[cell %02d/%02d] DONE in %.2fs", cell_index + 1, total, dt)
        for out in cell.get("outputs", []):
            t = out.get("output_type")
            if t == "stream":
                for line in out.get("text", "").rstrip().splitlines():
                    logger.info("    %s | %s", out.get("name", "out"), line)
            elif t in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "text/plain" in data and "image/png" not in data:
                    for line in str(data["text/plain"]).rstrip().splitlines()[:40]:
                        logger.info("    result | %s", line)
                elif "image/png" in data:
                    logger.info("    [figure rendered]")
            elif t == "error":
                logger.error("    ERROR %s: %s", out.get("ename"), out.get("evalue"))

    client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(folder)}},
                            allow_errors=False)
    client.on_cell_execute = on_start
    client.on_cell_executed = on_done
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    failed = False
    try:
        client.execute(cwd=str(folder))
    except CellExecutionError as e:
        failed = True
        logger.error("CELL FAILED:\n%s", str(e)[-3000:])
    except Exception:
        failed = True
        logger.error("RUNNER ERROR:\n%s", traceback.format_exc())

    nbformat.write(nb, nb_path)
    try:
        body, _ = HTMLExporter().from_notebook_node(nb)
        (folder / (nb_path.stem + ".html")).write_text(body, encoding="utf-8")
        logger.info("wrote %s.html", nb_path.stem)
    except Exception:
        logger.error("HTML export failed:\n%s", traceback.format_exc())

    logger.info("=== %s in %.1fs ===", "FAILED" if failed else "SUCCESS", time.perf_counter() - t_start)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
