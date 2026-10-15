"""Build the user guide PDF from the pages of the website.

The pages under documentation/guide/ are the only source of the user guide. This
script joins them in the order of the `nav` in mkdocs.yml and converts the
result to a PDF with pandoc and xelatex:

    python documentation/build_guide_pdf.py [output.pdf]

The default output is documentation/user_guide.pdf.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GUIDE = os.path.join(HERE, "guide")


def nav_pages():
    """Page files in the order of the nav in mkdocs.yml."""
    text = open(os.path.join(ROOT, "mkdocs.yml")).read()
    nav = text[text.index("\nnav:"):]
    return re.findall(r":\s*([\w./-]+\.md)\s*$", nav, flags=re.M)


def admonitions_to_quotes(text):
    """Turn a Material admonition into a block quote with a bold title."""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = re.match(r'^!!! \w+(?: "(.*)")?\s*$', lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        body, i = [], i + 1
        while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
            body.append(lines[i][4:])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        if m.group(1):
            out.append("> **%s.**" % m.group(1))
            out.append(">")
        out += [("> " + b).rstrip() for b in body]
        out.append("")
    return "\n".join(out)


def figure_code_to_pointer(text):
    """Replace a collapsed block of figure code by the path of the code.

    On the website the block opens on a click; the PDF cannot fold it, so
    it says where the scripts are instead of printing them.
    """
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = re.match(r'^\?\?\? example "Code of this figure \((.*)\)"\s*$', lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        i += 1
        while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
            i += 1
        out.append("The code of this figure is in `%s`." % m.group(1))
        out.append("")
    return "\n".join(out)


def local_links_to_text(text):
    """A link to another page of the site becomes its text."""
    return re.sub(r"\[([^\]]+)\]\((?:[\w-]+\.md)?(?:#[\w-]+)?\)", r"\1", text)


def light_figures_only(text):
    """Keep the light image of each #only-light/#only-dark pair.

    The website shows one of the two PNG files by theme; the PDF takes the
    light one, with the fragment stripped so pandoc finds the file.
    """
    text = re.sub(r"^!\[[^\]]*\]\([^)]*#only-dark\)[ \t]*\n?", "", text,
                  flags=re.M)
    return text.replace("#only-light)", ")")


def page_source(fn):
    text = light_figures_only(open(os.path.join(GUIDE, fn)).read())
    if fn == "index.md":
        text = re.sub(r"^# [^\n]*\n", "# Introduction\n", text, count=1)
    return local_links_to_text(admonitions_to_quotes(figure_code_to_pointer(text)))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "user_guide.pdf")
    md = "\n\n".join(page_source(fn) for fn in nav_pages())
    cmd = ["pandoc", "-f", "markdown", "-o", out, "--pdf-engine=xelatex",
           "--toc", "--toc-depth=1", "--number-sections",
           "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt",
           "-V", "colorlinks=true", "-V", "documentclass=article",
           "-M", "title=pyqula user guide",
           "--resource-path", GUIDE]  # images are img/... relative to documentation/guide/
    subprocess.run(cmd, input=md.encode(), check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
