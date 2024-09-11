from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

ext_modules = [
    Pybind11Extension(
        "toguz",
        ["toguz_board_wrapper.cpp"],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name="toguz",
    version="0.5.6",
    author="CockSucker",
    author_email="cocksucker@example.com",
    description="Toguzkumalak cpp logic wrapped with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
