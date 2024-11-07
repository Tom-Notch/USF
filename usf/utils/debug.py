#!/usr/bin/env python3
#
# Created on Sun Aug 24 2025 23:44:11
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
import inspect
import re
from typing import Any, Iterable

import pytorch_lightning as pl
import torch
from torch import nn


class FreezeBatchNorm(pl.Callback):
    """Lightning callback that freezes all BatchNorm layers at the start of training.

    Sets every ``BatchNorm1d`` / ``BatchNorm2d`` to eval mode and disables
    gradient computation for their parameters. Re-asserts the freeze at
    each epoch start in case training code toggles ``.train()`` on the module.
    """

    def _freeze_batchnorm(self, pl_module: pl.LightningModule) -> None:
        """Switch all BatchNorm layers to eval mode and freeze their parameters.

        Args:
            pl_module (pl.LightningModule): The Lightning module whose sub-modules are inspected.
        """
        for m in pl_module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Freeze BatchNorm layers when fitting begins.

        Args:
            trainer (pl.Trainer): The Lightning trainer.
            pl_module (pl.LightningModule): The model being trained.
        """
        self._freeze_batchnorm(pl_module)

    # Re-assert in case anything toggles modes later:
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Re-freeze BatchNorm layers at each epoch start.

        Args:
            trainer (pl.Trainer): The Lightning trainer.
            pl_module (pl.LightningModule): The model being trained.
        """
        self._freeze_batchnorm(pl_module)


class GradNormMonitor(pl.Callback):
    """Lightning callback that logs gradient norm statistics after each backward pass.

    Logs summary statistics (mean, max, median) and the top-k largest per-parameter
    gradient norms to the Lightning logger. Prints warnings when norms exceed a
    configurable threshold. NaN / Inf gradients are flagged and excluded from norm
    computation.
    """

    def __init__(
        self,
        patterns: list[str] | None = None,
        every_n_steps: int = 10,
        topk: int = 8,
        warn_thresh: float = 1e3,
    ) -> None:
        """Initialize a monitor callback for gradient normal

        Args:
            patterns (list[str] | None, optional): list of regex strings to select params by name (None = all). Defaults to None.
            every_n_steps (int, optional): log every N steps. Defaults to 10.
            topk (int, optional): also print top-k largest grad norms for quick eyeballing. Defaults to 8.
            warn_thresh (float, optional): print a warning if any grad norm exceeds this. Defaults to 1e3.
        """
        super().__init__()
        self.patterns = [re.compile(p) for p in (patterns or [])]
        self.every_n_steps = every_n_steps
        self.topk = topk
        self.warn_thresh = warn_thresh

    def _wanted(self, name: str) -> bool:
        """Check whether a parameter name matches any of the configured regex patterns.

        Args:
            name (str): Fully-qualified parameter name.

        Returns:
            bool: True if ``name`` matches any pattern, or if no patterns are configured.
        """
        if not self.patterns:
            return True
        return any(r.search(name) for r in self.patterns)

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute and log gradient norms after the backward pass.

        Only fires every ``every_n_steps`` global steps.

        Args:
            trainer (pl.Trainer): The Lightning trainer.
            pl_module (pl.LightningModule): The model whose gradients are inspected.
        """
        step = trainer.global_step
        if step % self.every_n_steps != 0:
            return

        norms = []
        for name, p in pl_module.named_parameters():
            if p.grad is None or not self._wanted(name):
                continue
            g = p.grad
            if torch.isnan(g).any() or torch.isinf(g).any():
                pl_module.print(f"[NaN/Inf GRAD] {name}")
                continue  # skip norm to avoid nan
            n = g.norm().item()
            norms.append((name, n))

        if not norms:
            return

        # log to Lightning (TensorBoard/W&B) as scalars
        # (avoid spamming: log summary stats + a few largest)
        values_only = [n for _, n in norms]
        log_dict = {
            "grads/count": len(values_only),
            "grads/mean": float(torch.tensor(values_only).mean().item()),
            "grads/max": float(max(values_only)),
            "grads/median": float(torch.tensor(values_only).median().item()),
        }
        pl_module.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)

        # top-k print for quick terminal diagnostics
        top = sorted(norms, key=lambda x: x[1], reverse=True)[: self.topk]
        if top:
            pl_module.print(
                f"[grad top-{self.topk}] " + ", ".join(f"{n}={v:.2e}" for n, v in top)
            )
        if any(v > self.warn_thresh for _, v in norms):
            pl_module.print(f"[warn] some grad norms > {self.warn_thresh:.1e}")


def call_with_filtered_kwargs(function: Any, provided: dict[str, Any]) -> Any:
    """Call ``function`` with only the kwargs it accepts; fall back to no-arg call on TypeError.

    If the function's signature includes ``**kwargs``, all ``provided`` entries are
    forwarded. Otherwise, only keys matching declared parameter names are passed.

    Args:
        function (Any): Callable to invoke.
        provided (dict[str, Any]): Keyword arguments to filter and pass.

    Returns:
        Any: Return value of ``function``.
    """
    try:
        sig = inspect.signature(function)
        # If function accepts **kwargs, just pass them all.
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return function(**provided)
        # Otherwise filter to names in the signature (ignore 'self').
        else:
            allowed = {k: v for k, v in provided.items() if k in sig.parameters}
            return function(**allowed) if allowed else function()
    except TypeError:
        # Fallback: no-arg call
        print(f"Calling {function} with no arguments since provided args errored out.")
        return function()


def safe_call(instance: Any, function_name: str, kwargs: dict[str, Any]) -> Any:
    """If `module` has callable `function_name`, call it with filtered kwargs.

    Args:
        instance (Any): Object to inspect for the named method.
        function_name (str): Name of the method to call.
        kwargs (dict[str, Any]): Keyword arguments to filter and pass.

    Returns:
        Any: result of the call, a string starting with "ERROR:" if the call raised, or None if function not present/callable.
    """
    function = getattr(instance, function_name, None)
    if not callable(function):
        return None
    try:
        return call_with_filtered_kwargs(function, kwargs)
    except Exception as e:
        return f"ERROR: {e}"


def recursive_call_functions(
    model: nn.Module,
    functions: Iterable[str],
    *,
    verbose: bool = True,
    kwargs: dict[str, Any] | None = None,
    per_function_kwargs: dict[str, dict[str, Any]] | None = None,
    include_missing: bool = False,
) -> dict[str, dict[str, Any]]:
    """Recursively traverse `model` and call each function in `functions` on submodules that implement it.

    Args:
        model (nn.Module): model to be inspected
        functions (Iterable[str]): collection of functions to run on each submodule that has them defined
        verbose (bool, optional): verbosity, True will print function called. defaults to True.
        kwargs (dict[str, Any] | None, optional): kwargs applied to all functions unless overridden below. defaults to None.
        per_function_kwargs (dict[str, dict[str, Any]] | None, optional): per-function kwargs. defaults to None.
                                                                   e.g. {"report_metrics": {"return_metrics": True}, "visualize_kernel": {"num_display_kernel": 2}}
        include_missing (bool, optional): include functions with None results in the sub-dict. defaults to False.

    Returns:
        dict ([str, dict[str, Any]]): { full_path: { function_name: result_or_error_or_None, ... }, ... }
    """
    results: dict[str, dict[str, Any]] = {}
    functions = tuple(functions)
    base_kwargs = dict(kwargs or {})
    per_function_kwargs = per_function_kwargs or {}

    with torch.no_grad():
        for path, module in model.named_modules():
            full_path = path if path else "<root>"
            submodule_result: dict[str, Any] = {}

            for function in functions:
                merged = {**base_kwargs, **per_function_kwargs.get(function, {})}
                result = safe_call(module, function, merged)
                if result is not None or include_missing:
                    submodule_result[function] = result
                    if verbose and result is not None:
                        print(
                            f"{full_path}.{function} successfully returned"
                            if not isinstance(result, str)
                            or not result.startswith("ERROR:")
                            else f"{full_path}.{function} errored out: {result}"
                        )

            if submodule_result:
                # only record modules where at least one func is present (or include_missing=True)
                # If include_missing=False and all values are None, skip
                if include_missing or any(
                    v is not None for v in submodule_result.values()
                ):
                    results[full_path] = submodule_result

    return results
