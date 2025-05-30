from __future__ import annotations

import inspect
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar, cast

from forge.agent.protocols import CommandProvider

from .parameter import CommandParameter

P = ParamSpec("P")
CO = TypeVar("CO")  # command output

_CP = TypeVar("_CP", bound=CommandProvider)


class Command(Generic[P, CO]):
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        names: list[str],
        description: str,
        method: Callable[Concatenate[_CP, P], CO],
        parameters: list[CommandParameter],
    ):
        # Check if all parameters are provided
        if not self._parameters_match(method, parameters):
            raise ValueError(
                f"Command {names[0]} has different parameters than provided schema"
            )
        self.names = names
        self.description = description
        # Method technically has a `self` parameter, but we can ignore that
        # since Python passes it internally.
        self.method = cast(Callable[P, CO], method)
        self.parameters = parameters

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    def _parameters_match(
        self, func: Callable, parameters: list[CommandParameter]
    ) -> bool:
        # Get the function's signature
        signature = inspect.signature(func)
        # Extract parameter names, ignoring 'self' for methods
        func_param_names = [
            param.name
            for param in signature.parameters.values()
            if param.name != "self"
        ]
        names = [param.name for param in parameters]
        # Check if sorted lists of names/keys are equal
        return sorted(func_param_names) == sorted(names)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> CO:
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: "
            + ("%s" if param.spec.required else "Optional[%s]")
            % (param.spec.type.value if param.spec.type else "Any")
            for param in self.parameters
        ]
        return (
            f"{self.names[0]}: {self.description.rstrip('.')}. "
            f"Params: ({', '.join(params)})"
        )

    def __get__(self, instance, owner):
        if instance is None:
            # Accessed on the class, not an instance
            return self
        # Bind the method to the instance
        return Command(
            self.names,
            self.description,
            self.method.__get__(instance, owner),
            self.parameters,
        )
