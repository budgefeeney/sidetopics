import pathlib

module_dir = pathlib.Path(__file__).absolute().parent
with open(module_dir / 'versionfile', 'r') as version_file:
    __version__ = version_file.read().strip()
