from setuptools import find_packages, setup
from setuptools.command.install import install
import os
import sys

BASEPATH = '.eDisGo'

if sys.version_info[:2] < (3, 7):
    error = (
        "eDisGo requires Python 3.7 or later (%d.%d detected)." % sys.version_info[:2]
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)

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
    version='0.1.1',
    packages=find_packages(),
    url='https://github.com/openego/eDisGo',
    license='GNU Affero General Public License v3.0',
    author='birgits, AnyaHe, gplssm, nesnoj, jaappedersen, Elias, boltbeard',
    author_email='anya.heider@rl-institut.de',
    description='A python package for distribution network analysis and optimization',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=[
        'demandlib',
        'networkx >= 2.5.0',
        'geopy >= 2.0.0',
        'pandas >= 1.2.0, < 1.3.0',
        'pyproj >= 3.0.0',
        'shapely >= 1.7.0',
        'pypsa >= 0.17.0',
        'pyomo >= 6.0',
        'multiprocess',
        'workalendar',
        'sqlalchemy < 1.4.0',
        'geoalchemy2 < 0.7.0',
        'egoio >= 0.4.7',
        'matplotlib >= 3.3.0',
        'pypower',
        'sklearn',
        'pydot',
        'Rtree',
    ],
    extras_require={
        'geoplot': ['geopandas >= 0.9.0', 'contextily', 'descartes'],
        'examples': ['jupyter'],
        'dev': ['pytest', 'sphinx_rtd_theme'],
        'full': ['geopandas >= 0.9.0', 'contextily', 'descartes', 'jupyter', 'pytest',
                 'sphinx_rtd_theme']
    },
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
