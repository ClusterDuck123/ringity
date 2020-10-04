import setuptools

with open("README.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ringity",
    version="0.0a17",
    author="Markus K. Youssef",
    author_email="mk.youssef@hotmail.com",
    description="ringity package",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/kiri93/ringity",
    packages=['ringity'],
    install_requires=['matplotlib', 'ripser', 'networkx', 'numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Topic :: Communications :: Email",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.6',
)
