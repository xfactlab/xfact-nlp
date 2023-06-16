from abc import ABC
from collections import defaultdict
import logging
from typing import cast, Callable, Type

logger = logging.getLogger(__name__)


class Registrable(ABC):
    class_registry = defaultdict(dict)

    @classmethod
    def register(cls,
                 name:str,
                 **kwargs):

        registry = Registrable.class_registry[cls]

        def register_cls(to_register):
            logger.debug(f"Registering class {to_register} with name {name}")
            if name in registry:
                raise RuntimeError(f"{name} is already registered")

            registry[name] = (to_register, kwargs)
            return to_register

        return register_cls

    @classmethod
    def _resolve(cls, name:str):
        registry = Registrable.class_registry[cls]
        if name in registry:
            subclass, args = registry[name]
            return subclass, args
        raise ValueError(f"Class '{name}' was not found in registry for {cls}")


    @classmethod
    def resolve(cls, name:str):
        clazz, _ = cls._resolve(name)
        return clazz

    @classmethod
    def get_constructor(cls, name :str):
        logger.debug(f"Getting constructor for {name}")

        subclass, args = cls._resolve(name)

        if not 'constructor' in args:
            logger.debug(f"No override function provided, using __init__")
            return cast(Type, subclass), args
        else:
            logger.debug(f"Using function {args['constructor']} as an override for constructor")
            return cast(Callable, getattr(subclass, args.pop('constructor'))), args

    @classmethod
    def init(cls, name, *args, **kwargs):
        cons, default_args = cls.get_constructor(name)
        default_args.update(kwargs)
        return cons(*args,**default_args)


if __name__ == "__main__":

    @Registrable.register("test")
    class AClass(Registrable):
        def __init__(self):
            print("I am init")

    print(Registrable.init("test"))

