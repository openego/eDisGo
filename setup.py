from setuptools import find_packages, setup
from setuptools.command.install import install
import os

BASEPATH='.eDisGo'


class InstallSetup(install):
    def run(self):
        self.create_edisgo_path()
        install.run(self)

    @staticmethod
    def create_edisgo_path():
        edisgo_path = os.path.join(os.path.expanduser('~'), BASEPATH)
        data_path = os.path.join(edisgo_path, 'data')

        if not os.path.isdir(edisgo_path):
            os.mkdir(edisgo_path)
        if not os.path.isdir(data_path):
            os.mkdir(data_path)


setup(
    name='eDisGo',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/openego/eDisGo',
    license='GNU Affero General Public License v3.0',
    author='gplssm, nesnoj',
    author_email='',
    description='A python package for distribution grid analysis and optimization',
    install_requires = [
        'ding0==0.1.2',
        'networkx >=1.11',
        'shapely >= 1.5.12, <= 1.5.12',
        'pandas >=0.20.3, <=0.20.3',
        'pypsa >=0.10.0, <=0.10.0'
    ],
    cmdclass={
      'install': InstallSetup}
)
