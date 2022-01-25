from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import call

def read_requirements(path):
    return list(Path(path).read_text().splitlines())


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


URL = 'https://github.com/arpytanshu/multivar-tsfc'


PROJECT_URLS = {
    'Bug Tracker': URL,
    'Documentation': URL,
    'Source Code': URL,
}

class CustomInstall(install):
    def run(self):
        install.run(self)
        call(['pip', 'install', 'torch==1.7.1+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])


setup(
      name='multivar',
      version="v0.1.0",
      description='streamlining multivariate timeseries forecasting research.',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      project_urls=PROJECT_URLS,
      url=URL,
      maintainer='Arpitanshu',
      maintainer_email='arpytanshu@gmail.com',
      license='Apache License 2.0',
      packages=find_packages(),
      cmdclass={
          'install': CustomInstall,
      },
      install_requires=read_requirements('requirements/main.txt'),
      zip_safe=False,
      python_requires='>=3.6',
      classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            ('Programming Language :: Python :: '
             'Implementation :: PyPy')
      ],
      keywords='multivariate timeseries forecasting pytorch'
)
