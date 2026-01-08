import hashlib
from typing import NamedTuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel

import analysisrun as ar
from analysisrun.scanner import Fields

plt.rcParams["font.family"] = "Arial"


def _stable_seed(s: str) -> int:
    # 64bitに収まる決定論的seed
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


class BenchParameters(BaseModel):
    # 計算コストを制御するパラメータ
    points_per_field: int = 200_000  # 各視野の系列長（描画用にはサンプリング）
    burn_in: int = 2_000  # 収束捨て
    repeats: int = 2  # 追加の繰り返し計算ループ回数（計算量スケール用）
    poly_degree: int = 512  # 多項モーメントの最高次数
    tau: int = 7  # 遅延座標
    plot_sample: int = 5_000  # 散布図の描画サンプル数/視野（多すぎると画像が重い）
    corr_len: Optional[int] = (
        None  # 相関行列用に使用する系列長（Noneならpoints_per_field）
    )


class BenchImageAnalysisResults(NamedTuple):
    activity_spots: Fields = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=ar.entity_filter("Activity Spots"),
    )


def _logistic_series(r: float, x0: float, burn_in: int, n: int) -> np.ndarray:
    # 1系列をループで生成（ここがCPU負荷の源泉）
    total = burn_in + n
    x = np.empty(total, dtype=np.float64)
    x[0] = x0
    for t in range(total - 1):
        x[t + 1] = r * x[t] * (1.0 - x[t])
    return x[burn_in:]


def _field_id(df_field: pd.DataFrame) -> int:
    if len(df_field) == 0:
        return -1
    v = df_field["MultiPointIndex"].iloc[0]
    try:
        return int(v)
    except Exception:
        return -1


def analyze(
    args: ar.AnalyzeArgs[BenchParameters, BenchImageAnalysisResults],
) -> pd.Series:
    params = args.params
    fields = args.image_analysis_results.activity_spots
    output = args.output

    series_by_field: list[np.ndarray] = []
    scatter_samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    field_ids: list[int] = []

    # lane名をseedにする（scanner.LanesでDataに分割済み）
    lane_name = fields.data_name
    base_seed = _stable_seed(str(lane_name))

    # lane中の視野ごとに系列を生成
    for df_field in fields:
        mpi = _field_id(df_field)
        # "Activity Spots"のみ入っている前提だが、空視野対策
        vals = (
            df_field["Value"].astype(float).to_numpy()
            if len(df_field)
            else np.array([0.5])
        )
        mu = float(np.mean(vals))
        sd = float(np.std(vals)) if np.std(vals) > 0 else 1.0

        # 決定論的seedからr, x0 を決める（chaos域に置く）
        seed = _stable_seed(f"{lane_name}:{mpi}:{mu:.6f}:{sd:.6f}") ^ base_seed
        rng = np.random.default_rng(seed)
        r = 3.90 + 0.09 * float(rng.random())
        x0 = float((np.tanh(mu / (abs(sd) + 1e-9)) + 1.0) * 0.5) % 1.0
        if x0 == 0.0:
            x0 = 0.123456789

        n = params.points_per_field
        tau = params.tau
        burn_in = params.burn_in

        seq = _logistic_series(r, x0, burn_in=burn_in, n=n + tau)
        # 遅延座標（散布図）
        X = seq[:-tau]
        Y = seq[tau:]
        # 描画サンプルの抽出（決定論的）
        m = len(X)
        take = min(params.plot_sample, m)
        idx = rng.choice(m, size=take, replace=False) if take < m else np.arange(m)
        scatter_samples.append((X[idx], Y[idx], mpi))

        # 追加の計算量：高次数モーメント
        # np.powerを次数ごとに回し、合計値を蓄積（結果自体は画像化に使わない）
        acc = 0.0
        for k in range(2, params.poly_degree + 1):
            acc += float(np.sum(np.power(seq, k, dtype=np.float64)))

        # 計算量のスケーリング：repeats回、決定論的に追加演算
        rep_acc = 0.0
        if params.repeats > 1:
            anchor = min(2000, len(seq))
            head = seq[:anchor]
            for _ in range(params.repeats - 1):
                # 内積＋三角関数合成などの軽い数値操作を繰返してCPUを回す
                rep_acc += float(
                    np.dot(head, head) + np.sin(head).sum() + np.cos(head).sum()
                )

        series_by_field.append(seq[: (params.corr_len or n)])
        field_ids.append(mpi)

    # 散布図の生成（視野ごとに色分け）
    fig1 = plt.figure(figsize=(8, 6), dpi=120)
    ax1 = fig1.add_subplot(111)
    cmap = plt.get_cmap("tab20")
    for Xs, Ys, mpi in scatter_samples:
        color = cmap((mpi - 1) % 20) if mpi > 0 else (0.3, 0.3, 0.3, 0.5)
        ax1.scatter(Xs, Ys, s=2, alpha=0.35, c=[color], label=f"F{mpi}")
    ax1.set_title(f"Lane {lane_name} - Delay embedding scatter")
    ax1.set_xlabel("x[t]")
    ax1.set_ylabel(f"x[t-{params.tau}]")
    # 凡例が多い時は省略気味に
    if 0 < len(field_ids) <= 12:
        ax1.legend(markerscale=4, fontsize=8, ncol=3, frameon=False)
    output(fig1, f"{lane_name}_scatter.png", "scatter", bbox_inches="tight")

    # 相関ヒートマップ（視野×視野）
    if len(series_by_field) >= 2:
        mat = np.vstack(series_by_field)  # shape: (num_fields, L)
        corr = np.corrcoef(mat)
        fig2 = plt.figure(figsize=(7, 6), dpi=120)
        ax2 = fig2.add_subplot(111)
        im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
        ax2.set_title(f"Lane {lane_name} - Field correlation")
        ax2.set_xlabel("Field index")
        ax2.set_ylabel("Field index")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        output(fig2, f"{lane_name}_corr.png", "heatmap", bbox_inches="tight")

    # 結果サマリ（ベンチ用のメタ情報）
    return pd.Series(
        dict(
            Lane=lane_name,
            NumFields=len(field_ids),
            PointsPerField=params.points_per_field,
            PolyDegree=params.poly_degree,
            Repeats=params.repeats,
            Tau=params.tau,
            Ok=True,
        )
    )


def postprocess(args: ar.PostprocessArgs[BenchParameters]):
    args.analysis_results["Postprocessed"] = True
    return args.analysis_results


test_dir = Path(__file__).parent / ".."
output_dir = test_dir / "testdata" / "results"

# 入力の読み取りと、Runnerの決定を行う
ctx = ar.read_context(
    BenchParameters,
    BenchImageAnalysisResults,
    manual_input=ar.ManualInput(
        BenchParameters(),
        {
            "activity_spots": test_dir / "testdata" / "image_analysis_result.csv",
        },
        sample_names=test_dir / "testdata" / "samples.csv",
    ),
    output_dir=output_dir,
)
print(ctx.mode)
results = ctx.run_analysis(analyze, postprocess)

results.to_csv(output_dir / "result.csv", index=False)

# FIXME: sequentialとparallelで画像の出力先が違う
