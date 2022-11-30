from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = f.read()

reqs = reqs.strip().split('\n')

install = [req for req in reqs if not req.startswith("git+git://")]
depends = [req.replace("git+git://", "git+http://") for req in reqs if req.startswith("git+git://")]


setup(
    name='xfact',
    version='0.0.0',
    author='James Thorne',
    author_email='james@jamesthorne.com',
    description='xfact',
    long_description="readme",
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=install,
    dependency_links=depends,
    package_dir={'':'src','xfact':'src/xfact'}
)