from setuptools import setup
from Cython.Build import cythonize

setup(
    name="This is a test",
    ext_modules=cythonize("ccentralities.pyx"),
    zip_safe=False,
)
