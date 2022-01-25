import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pepe',  
     version='1.0',
     author="Jack Featherstone",
     author_email="jdfeathe@ncsu.edu",
     description="Toolbox for working with photoelastic images",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Jfeatherstone/pepe",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
