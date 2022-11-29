from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

reqs = reqs.strip().split('\n')

install = [req for req in reqs if not req.startswith("git+git://")]
depends = [req.replace("git+git://", "git+http://") for req in reqs if req.startswith("git+git://")]


setup(
    name='xnlp',
    version='0.0.0',
    author='James Thorne',
    author_email='james@jamesthorne.com',
    description='xfact',
    long_description="readme",
    python_requires='>=3.8',
    packages=['xfact'],
    install_requires=install,
    dependency_links=depends,
    package_dir={'':'src'}
)