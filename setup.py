import os
from setuptools import setup
from distutils.command.build_ext import build_ext

class build_marker(build_ext):
    """Specialized Python source builder."""

    def run(self):
        cmd = "cd alpa/pipeline_parallel/xla_custom_call_marker; bash build.sh"
        return os.system(cmd)


setup(name="alpa",
      packages=["alpa"],
      cmdclass={"build_ext": build_marker})
