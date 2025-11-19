import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyqula",
    version="0.0.89",
    author="Jose Lado",
    author_email="joselado@aalto.fi",
    description="Python library for quantum lattice tight binding models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joselado/pyqula",
    project_urls={
        "Bug Tracker": "https://github.com/joselado/pyqula/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numba>=0.60.0","scipy","matplotlib","numpy"],
    python_requires=">=3.6",
)
