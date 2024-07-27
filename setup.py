'''
@author: Steven Tang <steven.tang@bytedance.com>
'''
from skbuild import setup

setup(
    name="MLInsight",
    version="1.0.1",
    packages=["mlinsight"],
    package_dir={"mlinsight": "src"},
    cmake_install_dir="src",
    cmake_args=['-DCMAKE_BUILD_TYPE=Debug']
)
