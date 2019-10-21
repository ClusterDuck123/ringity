import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ringity",
    version="0.0a3",
    author="Markus K. Youssef",
    author_email="mk.youssef@hotmail.com",
    description="ringity package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kiri93/ringity",
    packages=['ringity'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Topic :: Communications :: Email",
        "Development Status :: 2 - Pre-Alpha"
    ],
    python_requires='>=3.6',
)
