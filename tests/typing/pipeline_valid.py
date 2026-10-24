"""Public API examples checked by basedpyright, never executed."""

from typing import NamedTuple, assert_type

import pandas as pd
from pydantic import BaseModel

from analysisrun import (
    ExecutionContext,
    InitialPipeline,
    MapArgs,
    PipelineBuilder,
    PreprocessArgs,
    SynthesizeArgs,
    read_context,
)


class Params(BaseModel):
    threshold: int = 1


class Raw(NamedTuple):
    cells: pd.DataFrame


class Prepared(NamedTuple):
    cells: pd.DataFrame
    spots: pd.DataFrame


def prepare(args: PreprocessArgs[Params, Raw]) -> Prepared:
    assert_type(args.params, Params)
    assert_type(args.image_analysis_results, Raw)
    return Prepared(args.image_analysis_results.cells, pd.DataFrame())


def extract_features(args: MapArgs[Params, Prepared, None, None]) -> int:
    assert_type(args.value, None)
    assert_type(args.synthesized, None)
    assert_type(args.image_analysis_results, Prepared)
    return args.params.threshold


def evaluate_qc(args: SynthesizeArgs[Params, Prepared, int, None]) -> str:
    assert_type(args.values["a"], int)
    assert_type(args.image_analysis_results["a"], Prepared)
    return str(sum(args.values.values()))


def apply_qc(args: MapArgs[Params, Prepared, int, str]) -> float:
    return args.value + float(args.synthesized)


def summarize_results(args: SynthesizeArgs[Params, Prepared, float, str]) -> bool:
    return bool(args.values) and bool(args.synthesized)


def to_dataframe(args: SynthesizeArgs[Params, Prepared, float, bool]) -> pd.DataFrame:
    return pd.DataFrame({"value": list(args.values.values())})


def initial(args: SynthesizeArgs[Params, Raw, None, None]) -> pd.DataFrame:
    return pd.DataFrame()


def check() -> None:
    ctx = read_context(Params, Raw)
    assert_type(ctx, ExecutionContext[Params, Raw])
    assert_type(ctx.pipeline(), InitialPipeline[Params, Raw])
    assert_type(ctx.pipeline().finalize(initial), pd.DataFrame)
    prepared = ctx.pipeline().preprocess(prepare)
    assert_type(prepared, PipelineBuilder[Params, Prepared, None, None])
    mapped = prepared.map(extract_features)
    assert_type(mapped, PipelineBuilder[Params, Prepared, int, None])
    # Consecutive maps and syntheses update only their own state.
    mapped_twice = mapped.map(lambda a: str(a.value))
    assert_type(mapped_twice, PipelineBuilder[Params, Prepared, str, None])
    shared = mapped.synthesize(evaluate_qc)
    assert_type(shared, PipelineBuilder[Params, Prepared, int, str])
    shared_twice = shared.synthesize(lambda a: len(a.synthesized))
    assert_type(shared_twice, PipelineBuilder[Params, Prepared, int, int])
    assert_type(
        ctx.pipeline().synthesize(lambda a: 1), PipelineBuilder[Params, Raw, None, int]
    )
    result = (
        ctx.pipeline()
        .preprocess(prepare)
        .map(extract_features)
        .synthesize(evaluate_qc)
        .map(apply_qc)
        .synthesize(summarize_results)
        .finalize(to_dataframe)
    )
    assert_type(result, pd.DataFrame)
