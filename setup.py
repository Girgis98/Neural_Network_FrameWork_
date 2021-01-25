from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='FinalFlow',
  version='0.0.1',
  description='nn framework',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='thomas joseph',
  author_email='thomasartin@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='nnframework', 
  packages=find_packages(),
  install_requires=[''] 
)