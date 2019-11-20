from setuptools import find_packages, setup
from setuptools.command.install import install
import os

BASEPATH = '.eDisGo'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

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
    version='0.1.0dev',
    packages=find_packages(),
    url='https://github.com/openego/eDisGo',
    license='GNU Affero General Public License v3.0',
    author='gplssm, nesnoj, birgits, boltbeard, AnyaHe',
    author_email='anya.heider@rl-institut.de',
    description='A python package for distribution network analysis and optimization',
    long_description=read('README.md'),
    long_description_content_type='text/x-rst',
    install_requires=[
	    'demandlib',
        'networkx >= 2.0',
        'shapely >= 1.5.12, <= 1.6.3',
        'pandas',
        'pypsa >= 0.15.0',
        'pyproj >= 1.9.5.1, <= 1.9.5.1',
        'geopy >= 1.11.0, <= 1.11.0',
        'pyomo >= 5.5.0',
        'multiprocess',
        'workalendar',
        'oedialect',
        'geopandas',
        'descartes'
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
