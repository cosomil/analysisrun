"""Parent execution modes and streaming response lifecycle."""

import gzip
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import pandas as pd

from .env import ExecutionMode, ProcessRole, get_execution_mode, get_process_role
from .helper import cowsay
from .output import Output, _NullOutput, _save_image_bytes_to_dir
from .pipeable_io import ExitCodes, exit_with_error_streaming, redirect_stdout_to_stderr
from .pipeline import Inputs, Stage, execute_pipeline, execute_sequential
from .pipeline_worker import (
    ImageSink,
    MultiprocessMapExecutor,
    _add,
    _UniqueEntries,
    _write_streaming_result_entries,
    file_snapshot,
    run_worker,
    writing_tar,
)


def execute_entrypoint(
    stages: Sequence[Stage],
    *,
    entrypoint: Path,
    params: Any = None,
    image_analysis_results: Inputs | None = None,
    targets: Mapping[str, str] | None = None,
    output: Output | None = None,
    output_dir: Path | None = None,
    mode: ExecutionMode | None = None,
    process_role: ProcessRole | None = None,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
    stderr: IO[bytes] | None = None,
) -> pd.DataFrame:
    """Execution entrance shared by public builders and internal callers.

    Reconstruct stages before calling this in both parent and worker. Worker
    dispatch precedes input loading by the caller; no parent inputs are required.
    Notebook callers select notebook mode and provide a figure Output.
    """
    role = process_role or get_process_role()
    binary_stderr: IO[bytes] = stderr if stderr is not None else sys.stderr.buffer
    if role == "parallel-worker":
        worker_stdout = stdout if stdout is not None else sys.stdout.buffer
        with redirect_stdout_to_stderr(binary_stderr):
            status = run_worker(
                stages,
                stdin if stdin is not None else sys.stdin.buffer,
                worker_stdout,
            )
        sys.exit(status)
    mode = mode or get_execution_mode()
    if mode == "notebook":
        if image_analysis_results is None or targets is None:
            raise ValueError("notebook execution requires inputs and targets")
        if output is None and any(stage.kind == "map" for stage in stages):
            raise ValueError("notebook map execution requires an Output")
        try:
            return execute_sequential(
                stages,
                params=params,
                image_analysis_results=image_analysis_results,
                targets=targets,
                output=output if output is not None else _NullOutput(),
            )
        except Exception as exc:
            _report_processing_error(exc, binary_stderr)
            raise
    binary_stdout: IO[bytes] = stdout if stdout is not None else sys.stdout.buffer
    if mode not in {"interactive", "streaming"}:
        raise ValueError(f"unsupported pipeline mode: {mode!r}")
    if image_analysis_results is None or targets is None:
        raise ValueError("parallel execution requires inputs and targets")

    def run(on_image: ImageSink) -> pd.DataFrame:
        executor = MultiprocessMapExecutor(stages, entrypoint, on_image)
        return execute_pipeline(
            stages,
            params=params,
            image_analysis_results=image_analysis_results,
            targets=targets,
            output=output if output is not None else _NullOutput(),
            map_executor=executor,
            snapshot_factory=file_snapshot,
        )

    if mode == "interactive":
        if output_dir is None:
            raise ValueError("interactive requires output_dir")
        try:
            return run(
                lambda data_name, name, data, image_type: _save_image_bytes_to_dir(
                    name, data, output_dir
                )
            )
        except Exception as exc:
            _report_processing_error(exc, binary_stderr)
            raise SystemExit(ExitCodes.PROCESSING_ERROR.value) from exc
    with (
        gzip.GzipFile(fileobj=binary_stdout, mode="wb") as compressed,
        writing_tar(compressed) as raw_archive,
    ):
        archive = _UniqueEntries(raw_archive)

        def on_image(
            data_name: str, name: str, data: bytes, image_type: str | None
        ) -> None:
            _add(archive, f"{data_name}/images/{name}", data, image_type)
            compressed.flush()

        try:
            with redirect_stdout_to_stderr(binary_stderr):
                result = run(on_image)
            _write_streaming_result_entries(archive, result)
        except Exception as exc:  # noqa: BLE001 - every user/transport error must close the response
            exit_with_error_streaming(
                ExitCodes.PROCESSING_ERROR, str(exc), archive, binary_stderr, exc
            )
    sys.exit(0)


def _report_processing_error(exc: Exception, stderr: IO[bytes]) -> None:
    stderr.write("".join(traceback.format_exception(exc)).encode())
    context = "\n".join(getattr(exc, "__notes__", ()))
    message = f"解析処理に失敗しました。{context}\n担当者にお問い合わせください。"
    stderr.write((cowsay(message) + "\n").encode())
    stderr.flush()
