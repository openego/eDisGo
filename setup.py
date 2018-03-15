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
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/openego/eDisGo',
    license='GNU Affero General Public License v3.0',
    author='gplssm, nesnoj, birgits',
    author_email='',
    description='A python package for distribution grid analysis and optimization',
    install_requires = [
        'ding0==0.1.4',
        'networkx >=1.11, <2.0 ',
        'shapely >= 1.5.12, <= 1.6.3',
        'pandas >=0.20.3, <=0.20.3',
        'pypsa >=0.11.0, <=0.11.0',
        'pyproj >= 1.9.5.1, <= 1.9.5.1',
        'geopy >= 1.11.0, <= 1.11.0',
        'workalendar',
        'demandlib'
    ],
    package_data={
        'edisgo': [
            os.path.join('config', 'config_system'),
            os.path.join('config', '*.cfg'),
            os.path.join('equipment', '*.csv')]
    },
    cmdclass={
        'install': InstallSetup},
        entry_points={
            'console_scripts': ['edisgo_run = edisgo.tools.edisgo_run:edisgo_run']
        }
)
