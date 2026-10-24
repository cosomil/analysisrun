"""Public context construction and Pipeline execution."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ValidationError

from .env import (
    ExecutionMode,
    ProcessRole,
    get_entrypoint,
    get_execution_mode,
    get_process_role,
)
from .input import (
    InputModel,
    ManualInput,
    _build_streaming_input_schema,
    _derive_output_dir,
    _get_image_analysis_specs,
    _maybe_load_json,
    _prepare_inputs,
    create_image_analysis_results_input_model,
)
from .interactive import scan_model_input
from .output import Output, _FileOutput, _NotebookOutput, _NullOutput
from .pipeable_io import ExitCodes, exit_with_error
from .tar import read_tar_as_dict
from .typing import NamedTupleLike

if TYPE_CHECKING:
    from .pipeline import InitialPipeline, Stage


@dataclass
class ExecutionContext[P: BaseModel, I: NamedTupleLike[pd.DataFrame]]:
    """Prepared inputs belong to the parent; workers only reconstruct definitions."""

    params: P
    image_analysis_results: I | None
    targets: dict[str, str] | None
    output: Output
    mode: ExecutionMode | None
    process_role: ProcessRole
    entrypoint: Path | None
    output_dir: Path | None
    stdin: IO[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]

    def pipeline(self) -> "InitialPipeline[P, I]":
        from .pipeline import InitialPipeline

        return InitialPipeline(self)

    def _execute_pipeline(self, stages: tuple["Stage", ...]) -> pd.DataFrame:
        from .pipeline_entrypoint import execute_entrypoint

        return execute_entrypoint(
            stages,
            entrypoint=self.entrypoint or Path("."),
            params=self.params,
            image_analysis_results=self.image_analysis_results,
            targets=self.targets,
            output=self.output,
            output_dir=self.output_dir,
            mode=self.mode,
            process_role=self.process_role,
            stdin=self.stdin,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def read_context[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    params: type[Params],
    image_analysis_results: type[ImageAnalysisResults],
    manual_input: ManualInput[Params] | None = None,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
    output_dir: str | Path | None = None,
) -> ExecutionContext[Params, ImageAnalysisResults]:
    """Read and normalize parent input; workers skip original input entirely."""
    _stdin: IO[bytes] = stdin or sys.stdin.buffer
    _stdout: IO[bytes] = stdout or sys.stdout.buffer
    _stderr: IO[bytes] = sys.stderr.buffer
    role = get_process_role()
    if role == "parallel-worker":
        # Rebuild only the definition. Actual params/inputs arrive via the trusted
        # worker protocol at finalize; do not read or validate parent input here.
        return ExecutionContext(
            params=None,  # type: ignore[arg-type]
            image_analysis_results=None,
            targets=None,
            output=_NullOutput(),
            mode=None,
            process_role=role,
            entrypoint=None,
            output_dir=None,
            stdin=_stdin,
            stdout=_stdout,
            stderr=_stderr,
        )

    try:
        mode = get_execution_mode()
    except ValueError as exc:
        exit_with_error(ExitCodes.INVALID_USAGE, str(exc), _stdout, _stderr)

    try:
        specs = _get_image_analysis_specs(image_analysis_results)
        image_analysis_results_input_model = create_image_analysis_results_input_model(
            image_analysis_results
        )
    except Exception as exc:  # noqa: BLE001
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            "ImageAnalysisResultsの定義が不正です。各フィールドにimage_analysis_result_spec(...)を設定してください。",
            _stdout,
            _stderr,
            exc,
        )
    RuntimeInputModel = InputModel[params, image_analysis_results_input_model]
    output_dir_path = Path(output_dir) if output_dir else None

    if mode == "showschema":
        schema = _build_streaming_input_schema(params, image_analysis_results)
        _stdout.write(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        _stdout.flush()
        raise SystemExit(0)

    if mode == "streaming":
        try:
            tar_dict = read_tar_as_dict(_stdin)
            tar_dict["params"] = _maybe_load_json(tar_dict.get("params"))
            runtime_input = RuntimeInputModel(**tar_dict)
        except Exception as exc:  # noqa: BLE001
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "入力データの読み込みに失敗しました。入力形式を確認してください。",
                _stdout,
                _stderr,
                exc,
            )
    elif manual_input is not None:
        try:
            iar_input = image_analysis_results_input_model(
                **manual_input.image_analysis_results
            )
            runtime_input = RuntimeInputModel(
                image_analysis_results=iar_input,
                sample_names=manual_input.sample_names,  # type: ignore
                params=manual_input.params,
            )
        except ValidationError as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "manual_inputの形式が正しくありません。",
                _stdout,
                _stderr,
                exc,
            )
    else:
        if mode == "notebook":
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "Jupyter notebook環境ではmanual_inputの指定が必須です。",
                _stdout,
                _stderr,
            )

        # CLI等ではインタラクティブな入力を通じてRuntimeInputModelを組み立てる。
        runtime_input = scan_model_input(RuntimeInputModel)

    try:
        pipeline_inputs, targets = _prepare_inputs(
            image_analysis_results,
            runtime_input.image_analysis_results,
            runtime_input.sample_names,
            specs,
        )

    except Exception as exc:  # noqa: BLE001 - input boundary reports user cleansing errors
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            "入力データの読み込みに失敗しました。入力形式を確認してください。",
            _stdout,
            _stderr,
            exc,
        )

    entrypoint = None
    output_impl: Output = _NullOutput()
    if mode == "notebook":
        output_impl = _NotebookOutput()
    elif mode == "interactive":
        if output_dir_path is None:
            output_dir_path = _derive_output_dir(runtime_input.image_analysis_results)
        output_impl = _FileOutput(output_dir_path)
    if mode != "notebook":
        entrypoint = get_entrypoint()
        if entrypoint is None:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "エントリーポイントとなるスクリプトのパス取得に失敗したため、並列実行させることができません。",
                _stdout,
                _stderr,
            )
    return ExecutionContext(
        params=runtime_input.params,
        image_analysis_results=pipeline_inputs,
        targets=targets,
        output=output_impl,
        mode=mode,
        process_role=role,
        entrypoint=entrypoint,
        output_dir=output_dir_path,
        stdin=_stdin,
        stdout=_stdout,
        stderr=_stderr,
    )
