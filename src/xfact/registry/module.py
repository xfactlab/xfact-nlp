import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)



def import_submodules(package_name: str) -> None:
    importlib.invalidate_caches()
    logger.info(f"Import packages from {package_name}")

    module = importlib.import_module(package_name)
    path = getattr(module, '__path__', [])
    path_string = '' if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)
        logger.info(f"Import subpackage {subpackage}")

