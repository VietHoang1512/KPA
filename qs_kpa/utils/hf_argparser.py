import dataclasses
import json
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, NewType, Tuple, Union

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


class HfArgumentParser(ArgumentParser):

    """This subclass of `argparse.ArgumentParser` uses type hints on
    dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In
    particular, you can add more (non-dataclass backed) arguments to the
    parser after initialization and you'll get the output back after
    parsing as an additional namespace.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances
                with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for x in (int, float, str):
                if typestring == f"typing.Union[{x.__name__}, NoneType]":
                    field.type = x
            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool:
                kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no-{field.name}"
                    kwargs["dest"] = field.name
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True
    ) -> Tuple[DataClass, ...]:

        if look_for_args_file and len(sys.argv):
            args_file = Path(sys.argv[0]).with_suffix(".args")
            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]

        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,

        instead loading a json file and populating the dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)


if __name__ == "__main__":
    from qs_kpa.baselines.data_argument import DataArguments
    from qs_kpa.baselines.model_argument import ModelArguments
    from qs_kpa.train_utils.training_argument import TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)
    print(training_args)
    print("DONE")
