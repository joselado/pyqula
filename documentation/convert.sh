pandoc user_guide.md -o user_guide.pdf --toc --toc-depth=2 --pdf-engine=xelatex
# xelatex, not the default pdflatex: the guide contains unicode (the Greek
# letters of the physics prose, e.g. the Gamma point) that pdflatex cannot
# typeset, and the build simply failed on the first one it reached
