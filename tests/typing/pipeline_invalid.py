"""Every marked line must produce the specified diagnostic."""

from __future__ import annotations

from analysisrun import ExecutionContext, MapArgs, SynthesizeArgs
from tests.typing.pipeline_valid import (
    Params,
    Prepared,
    Raw,
    evaluate_qc,
    extract_features,
    prepare,
)


def wrong_value(args: MapArgs[Params, Prepared, str, None]) -> int:
    return 1


def wrong_shared(args: MapArgs[Params, Prepared, int, bool]) -> int:
    return 1


def wrong_inputs(args: MapArgs[Params, Raw, int, None]) -> int:
    return 1


def wrong_params(args: MapArgs[OtherParams, Prepared, int, None]) -> int:
    return 1


class OtherParams(Params):
    other: int = 2


def wrong_synthesis(args: SynthesizeArgs[Params, Prepared, str, None]) -> int:
    return 1


def check(ctx: ExecutionContext[Params, Raw]) -> None:
    prepared = ctx.pipeline().preprocess(prepare)
    mapped = prepared.map(extract_features)
    prepared.preprocess(prepare)  # error: reportAttributeAccessIssue
    mapped.preprocess(prepare)  # error: reportAttributeAccessIssue
    shared = ctx.pipeline().synthesize(lambda a: 1)
    shared.preprocess(prepare)  # error: reportAttributeAccessIssue
    mapped.map(wrong_value)  # error: reportArgumentType
    mapped.map(wrong_inputs)  # error: reportArgumentType
    mapped.map(wrong_params)  # error: reportArgumentType
    mapped.synthesize(wrong_synthesis)  # error: reportArgumentType
    mapped.synthesize(evaluate_qc).map(wrong_shared)  # error: reportArgumentType
    mapped.finalize(lambda a: 1)  # error: reportArgumentType
    ctx.pipeline().preprocess(lambda a: 1)  # error: reportArgumentType
