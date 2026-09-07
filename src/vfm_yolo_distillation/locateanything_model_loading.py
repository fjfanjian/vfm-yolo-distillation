from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import FunctionType
from typing import Any


def load_locateanything_model(model_id: str, dtype: Any) -> Any:
    from transformers import AutoConfig, AutoModel

    patch_dynamic_cache_legacy_compat()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    patch_locateanything_config_compat(model_id, config)
    model_class = _remote_auto_model_class(model_id, config)
    if model_class is None:
        return AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    patch_remote_package_compat(model_class)
    return model_class.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def patch_remote_package_compat(model_class: type[Any]) -> None:
    patch_remote_attn_compat(model_class)
    patch_missing_all_tied_weights_keys(model_class)
    patch_tied_weights_compat(model_class)
    package_prefix = model_class.__module__.rsplit(".", maxsplit=1)[0]
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith(package_prefix):
            continue
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if getattr(candidate, "__module__", "").startswith(package_prefix):
                patch_remote_attn_compat(candidate)
                patch_missing_all_tied_weights_keys(candidate)
                patch_tied_weights_compat(candidate)


def patch_locateanything_config_compat(model_id: str, config: Any) -> None:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    raw_text_config = _raw_text_config(model_id)
    for key in ("rope_theta", "rope_scaling"):
        if hasattr(text_config, key) or key not in raw_text_config:
            continue
        setattr(text_config, key, raw_text_config[key])


def patch_dynamic_cache_legacy_compat(cache_class: type[Any] | None = None) -> None:
    if cache_class is None:
        from transformers.cache_utils import DynamicCache

        cache_class = DynamicCache
    if not hasattr(cache_class, "to_legacy_cache"):

        def to_legacy_cache(self: Any) -> Any:
            return self

        cache_class.to_legacy_cache = to_legacy_cache

    if not hasattr(cache_class, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls: type[Any], past_key_values: Any = None) -> Any:
            if isinstance(past_key_values, cls):
                return past_key_values
            if past_key_values is None:
                return cls()
            return cls(past_key_values)

        cache_class.from_legacy_cache = from_legacy_cache


def patch_remote_attn_compat(model_class: type[Any]) -> None:
    method_name = "_check_and_adjust_attn_implementation"
    for candidate in model_class.mro():
        descriptor = candidate.__dict__.get(method_name)
        if descriptor is None:
            continue
        function = _unwrap_method_descriptor(descriptor)
        if _accepts_allow_all_kernels(function):
            return

        def compat(
            self: Any,
            *args: Any,
            __function: FunctionType = function,
            **kwargs: Any,
        ) -> Any:
            kwargs.pop("allow_all_kernels", None)
            return __function(self, *args, **kwargs)

        setattr(candidate, method_name, _rewrap_method_descriptor(descriptor, compat))
        return


def patch_tied_weights_compat(model_class: type[Any]) -> None:
    keys = getattr(model_class, "_tied_weights_keys", None)
    if not isinstance(keys, list):
        return
    original = getattr(model_class, "get_expanded_tied_weights_keys", None)
    if getattr(original, "_locateanything_tied_weights_compat", False):
        return

    def get_expanded_tied_weights_keys(
        self: Any,
        all_submodels: bool = True,
        __original: Any = original,
    ) -> Any:
        current_keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(current_keys, list):
            return current_keys
        if __original is None:
            return current_keys
        return __original(self, all_submodels=all_submodels)

    get_expanded_tied_weights_keys._locateanything_tied_weights_compat = True
    model_class.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys


def patch_missing_all_tied_weights_keys(model_class: type[Any]) -> None:
    original_init = model_class.__dict__.get("__init__")
    if original_init is None:
        return
    if getattr(original_init, "_locateanything_all_tied_weights_compat", False):
        return

    def init_compat(
        self: Any,
        *args: Any,
        __original_init: Any = original_init,
        **kwargs: Any,
    ) -> None:
        __original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    init_compat._locateanything_all_tied_weights_compat = True
    model_class.__init__ = init_compat


def _remote_auto_model_class(model_id: str, config: Any) -> type[Any] | None:
    auto_map = getattr(config, "auto_map", None)
    if not isinstance(auto_map, dict):
        return None
    class_ref = auto_map.get("AutoModel")
    if not isinstance(class_ref, str):
        return None
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(
        class_ref,
        model_id,
        trust_remote_code=True,
    )


def _unwrap_method_descriptor(descriptor: Any) -> FunctionType:
    if isinstance(descriptor, (classmethod, staticmethod)):
        return descriptor.__func__
    return descriptor


def _rewrap_method_descriptor(original: Any, replacement: FunctionType) -> Any:
    if isinstance(original, classmethod):
        return classmethod(replacement)
    if isinstance(original, staticmethod):
        return staticmethod(replacement)
    return replacement


def _accepts_allow_all_kernels(function: FunctionType) -> bool:
    signature = inspect.signature(function)
    for parameter in signature.parameters.values():
        if parameter.name == "allow_all_kernels":
            return True
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _raw_text_config(model_id: str) -> dict[str, Any]:
    config_path = Path(model_id) / "config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = payload.get("text_config", {})
    if isinstance(text_config, dict):
        return text_config
    return {}
