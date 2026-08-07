"""Render benchmarks/results/*.json into a LaTeX report (report.tex), with
one table + one log-log wall-time-vs-size plot per case, and compile it to
PDF with latexmk (falling back to pdflatex) if either is available.
Degrades gracefully -- writes report.tex and prints a warning -- on
machines without a LaTeX toolchain, since neither is a repo dependency.
"""
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.harness import load_records


def _escape(s):
    return str(s).replace("_", r"\_")


def _plot_case(results_dir, case_name, records):
    methods = sorted(set(r["method"] for r in records))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for method in methods:
        pts = sorted((r["size"], r["t_warm"]) for r in records if r["method"] == method)
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, marker="o", label=method)
    ax.set_xlabel("size")
    ax.set_ylabel("warm wall time [s]")
    ax.set_title(case_name.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    plot_name = f"{case_name}.pdf"
    fig.savefig(os.path.join(results_dir, plot_name))
    plt.close(fig)
    return plot_name


def _table(records):
    header = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "method & size & cold [s] & warm [s] & reldiff vs ref \\\\\n\\midrule\n"
    )
    rows = []
    for r in sorted(records, key=lambda r: (r["method"], r["size"])):
        reldiff = r.get("reldiff")
        reldiff_s = f"{reldiff:.2e}" if reldiff is not None else "--"
        rows.append(
            f"{_escape(r['method'])} & {r['size']} & {r['t_cold']:.4g} & "
            f"{r['t_warm']:.4g} & {reldiff_s} \\\\"
        )
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def _machine_block(machine):
    if not machine:
        return ""
    lines = []
    for key in ("platform", "processor", "cpu_count", "numpy", "scipy", "numba", "jax",
                "jax_devices", "pyqula_cores"):
        if key in machine:
            lines.append(f"{_escape(key)}: {_escape(machine[key])} \\\\")
    return "\n".join(lines)


def build_report(results_dir, case_names, out_name="report"):
    machine = None
    sections = []
    for case_name in case_names:
        data = load_records(results_dir, case_name)
        if machine is None:
            machine = data["machine"]
        records = data["records"]
        plot_name = _plot_case(results_dir, case_name, records)
        table = _table(records)
        sections.append(
            "\\section{%s}\n\n%s\n\n\\begin{center}\n"
            "\\includegraphics[width=0.7\\textwidth]{%s}\n\\end{center}\n"
            % (_escape(case_name), table, plot_name)
        )

    tex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=2.5cm]{geometry}
\title{pyqula benchmark report}
\author{}
\date{\today}
\begin{document}
\maketitle
\section*{Machine}
%s

%s
\end{document}
""" % (_machine_block(machine), "\n".join(sections))

    tex_path = os.path.join(results_dir, f"{out_name}.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    return tex_path


def compile_report(tex_path):
    """Compile tex_path to PDF in place. Returns the PDF path, or None if no
    LaTeX toolchain is available (a printed warning explains why)."""
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    results_dir = os.path.dirname(tex_path) or "."
    tex_name = os.path.basename(tex_path)
    if latexmk:
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_name]
    elif pdflatex:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name]
    else:
        print(f"No LaTeX toolchain found (latexmk/pdflatex); leaving {tex_path} uncompiled.")
        return None
    subprocess.run(cmd, cwd=results_dir, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    return pdf_path if os.path.exists(pdf_path) else None
