from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

type ConfigValue = int | str | Sequence[int] | Sequence[str]


@dataclass(frozen=True, slots=True)
class LayerSpec:
    layer: int
    dim: int


@dataclass(frozen=True, slots=True)
class MultiLayerSettings:
    layers: tuple[LayerSpec, ...]


@dataclass(frozen=True, slots=True)
class MultiLayerConfigError(RuntimeError):
    message: str

    @classmethod
    def length_mismatch(cls) -> MultiLayerConfigError:
        return cls("student_layers and student_dims must have the same length")

    @classmethod
    def empty_layers(cls) -> MultiLayerConfigError:
        return cls("student_layers must not be empty")

    @classmethod
    def invalid_sequence(cls, key: str) -> MultiLayerConfigError:
        return cls(f"{key} must be an int, string, or integer list")

    @classmethod
    def missing_required(cls, key: str) -> MultiLayerConfigError:
        return cls(f"{key} is required")

    @classmethod
    def invalid_integer(cls, key: str) -> MultiLayerConfigError:
        return cls(f"{key} must be an integer")

    @classmethod
    def non_integer_value(cls, key: str, value: str) -> MultiLayerConfigError:
        return cls(f"{key} contains a non-integer value: {value}")

    @classmethod
    def invalid_items(cls, key: str) -> MultiLayerConfigError:
        return cls(f"{key} must contain only integers")

    def __str__(self) -> str:
        return self.message


def parse_multilayer_settings(objectness: Mapping[str, ConfigValue]) -> MultiLayerSettings:
    layers = _optional_int_tuple(objectness.get("student_layers"), "student_layers")
    dims = _optional_int_tuple(objectness.get("student_dims"), "student_dims")
    if not layers and not dims:
        return MultiLayerSettings(
            layers=(
                LayerSpec(
                    layer=_required_int(objectness.get("student_layer"), "student_layer"),
                    dim=_required_int(objectness.get("student_dim"), "student_dim"),
                ),
            )
        )
    if len(layers) != len(dims):
        raise MultiLayerConfigError.length_mismatch()
    if not layers:
        raise MultiLayerConfigError.empty_layers()
    return MultiLayerSettings(
        layers=tuple(
            LayerSpec(layer=layer, dim=dim) for layer, dim in zip(layers, dims, strict=True)
        )
    )


def _optional_int_tuple(value: ConfigValue | None, key: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, str):
        return tuple(_coerce_int(part.strip(), key) for part in value.replace("+", ",").split(","))
    if isinstance(value, Sequence):
        return tuple(_coerce_int(item, key) for item in value)
    raise MultiLayerConfigError.invalid_sequence(key)


def _required_int(value: ConfigValue | None, key: str) -> int:
    if value is None:
        raise MultiLayerConfigError.missing_required(key)
    return _coerce_int(value, key)


def _coerce_int(value: ConfigValue, key: str) -> int:
    if isinstance(value, bool):
        raise MultiLayerConfigError.invalid_integer(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise MultiLayerConfigError.non_integer_value(key, value) from exc
    raise MultiLayerConfigError.invalid_items(key)
