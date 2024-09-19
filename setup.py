import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pepe-granular',  
     version='1.2.7',
     author="Jack Featherstone",
     author_email="jack.featherstone@oist.jp",
     license='MIT',
     url='https://jfeatherstone.github.io/pepe/pepe',
     description="Image analysis toolbox tailored towards granular photoelastic images",
     long_description=long_description,
     long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.11",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
