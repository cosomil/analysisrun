# Pipeline の状態・転送契約（#82）

対象: [#82](https://github.com/cosomil/analysisrun/issues/82)、
[実装・検証仕様](https://github.com/cosomil/analysisrun/issues/82#issuecomment-6283053369)。
[#81 の構築モデル](pipeline.md) を前提にする。連続 map の実行範囲とバリアについては
本書が #81 の単一 `stage_index` 案を更新する。
本番 Executor は実装せず、`tests/pipeline_transfer_probe.py` と
`tests/test_pipeline_transfer_probe.py` に独立実験を置く。既存 API は変更しない。

## 状態と遷移

親が、preprocess 後の `image_analysis_results` のスナップショット、レーン別
`values`、共有 `synthesized` を所有する。元の ExecutionContext 入力、params、
固定の対象レーン（data_name / sample_name / 順序）は別に扱う。新 API に extra はない。

| 境界 | 場所 | 入力 | 更新・出力 | 次へ進める条件 |
| --- | --- | --- | --- | --- |
| finalize による開始 | 親 | 元入力、固定の対象集合 | 各 value と synthesized を None に初期化 | 定義検証済み |
| preprocess（省略可） | 親で最大1回 | コピーした元入力・params、対象集合 | 戻り値の NamedTupleLike を入力として確定 | レーン分割・全スナップショットの保存成功 |
| Map Segment | 子（Notebook は逐次） | 各レーン入力、params、value、synthesized | レーン完了ごとに value を親で置換。画像は随時転送 | 全子の終了コード0、正常 TAR、全担当 value 受信 |
| synthesize | 親 | 復元した全入力、values、params、synthesized | 戻り値で synthesized のみ置換 | 正常 return |
| finalize の最終処理 | 親 | synthesize と同じ全体引数 | 最終 pd.DataFrame | 正常 return |

map / synthesize の戻り値は None も正当であり、以前の値へ暗黙に戻さない。
preprocess は値の初期化ではなく入力型 I → J の変換。その結果を再計算せず、全後続
ステージの入力の正本にする。ゼロレーンでは map は呼ばず、全体処理には空 Mapping を渡す。

連続する map は一つの **Map Segment** にまとめ、同一レーンについて同じ子で順に
実行する。一子に複数レーンを割り当ててよい。中間 value は Python オブジェクトを
直接渡し、pickle 化しない。value の破壊的変更を許容し、次の value は必ず戻り値を採用する。
入力と value の参照共有自体は検査しない。

values のステージ単位一括コミットは行わない。`value.pkl` 到着時点で親の values を
更新し、画像も到着時点で処理する。失敗しても受信済みの画像・値をロールバックしない。
ただし **Segment 間のバリア**は必須であり、一子でも失敗すれば次の synthesize /
Map Segment / finalize を実行しない。連続 map の途中に全レーンのバリアは置かない。

## 所有権・コピー・一時ファイル

| 対象 | 親の処理 | 子の各レーン・各 map 呼び出し |
| --- | --- | --- |
| 入力 | 各 synthesize/finalize 時に全レーンを一括復元し外側を MappingProxyType にする | 担当レーンの NamedTuple インスタンス全体を毎回ファイルから復元 |
| params | preprocess を含む各呼び出しに独立したコピー | 一子に一度転送した pickle バイト列を毎回復元 |
| synthesized | synthesize/finalize 間で同じ参照。入力として変更しない契約 | 一子に一度転送した pickle バイト列を毎回復元 |
| values | 外側のみ MappingProxyType。要素はコピーせず変更禁止の契約 | 担当 value を Segment 開始時に一度復元。連続 map 間は戻り値を直接渡す |

synthesize が values の要素をそのまま synthesized として返してよい。Mapping の保護は
キーの追加・削除・置換のみで、要素の破壊的変更を検出・防止しない。履歴や実行済みフラグは持たない。

マルチプロセスの寿命:

1. finalize 到達時に開始し、preprocess へは元入力のコピーを渡す。
2. preprocess 完了直後（省略時は実行開始時）、レーンごとの NamedTuple インスタンス
   **全体**を `pickle.HIGHEST_PROTOCOL` で TemporaryDirectory に保存する。
   先行する synthesize が入力を変更しても影響しないよう、最初の map まで遅延しない。
3. 全レーンの write / close 成功後に子を起動する。書込中の並行参照がないため
   os.replace は不要。子は rb で読み、書き戻さない。chmod 等の強制保護も不要。
4. 各 map の直前に再読込する。連続 map、同じ子の別レーンのいずれも入力の変更を
   伝播させない。親も同じファイルから毎回全レーンを復元し、遅延ロードや常駐キャッシュは使わない。
5. 親が正常・異常とも全子の終了・回収を確認してから TemporaryDirectory を削除する。
   多数の子の停止・回収の製品実装は #84。

親が preprocess 後の全体 DataFrame を保持し続ける必要はない。入力はレーン別
ファイルを一度作り、以降はパスだけ送るため、巨大な不変入力を Segment ごとに再送しない。
毎 map のディスク読込と復元コストを受け入れ、mmap や遅延ロードの複雑さを導入しない。
値と params / synthesized は Segment ごとに送る。ここでの方式比較は設計判断であり、
大規模な速度・メモリの優劣を実測したものではない。

Notebook は pickle 可能性を要求しない。前処理に元入力のコピーを渡し、変換後入力から
内部スナップショットを一度コピーする。毎 map は担当レーン、synthesize/finalize は
全レーンの各 DataFrame を `copy(deep=True)` し NamedTuple 型を保って再構築する。
object 列内部のオブジェクトまでの完全分離は保証しない。params は各呼び出しで独立した
コピー、共有値の扱いは上表の親と同じ。

対象集合と data_name / sample_name / 順序は実行中不変。各入力 DataFrame は対象全レーンへ
分割可能でなければならず、欠損・改名を map 前に拒否する。対象外の行から新レーンを作らない。
行の削減、列追加、値変換は許容し、preprocess 後の cleansing は行わない。

## 転送 v1

同一の信頼済み Python 環境で実行する。pickle は任意コード実行が可能なため、外部の
信頼できない TAR / pkl を読み込むインターフェースとして公開しない。
異なる Python バージョン間の互換性は対象外。`pickle.HIGHEST_PROTOCOL` を使用し、
事前に pickle できるか検査しない。実際の失敗をデータ種別・レーン・ステージを付けて通知する。

`protocol_version = 1` は Manifest の version と独立。不一致はステージ実行前に拒否し、
旧形式から自動変換しない。子は全 Manifest（preprocess / synthesize / finalize も含む）を
再構築して全文比較する。実行範囲は全定義の **0 始まり半開区間 [stage_start, stage_end)**。
bool を除く整数、非空、範囲内、全要素 map を検証してから実行する。
#81 の単一 stage_index はこの範囲に置き換える。エラーの stage_index は絶対位置のまま。

親 → 子（TAR/stdin）:

```text
request.json
params.pkl
synthesized.pkl
lanes/0/value.pkl
lanes/1/value.pkl
...
```

`request.json` の概念スキーマ:

```json
{
  "protocol_version": 1,
  "manifest": "canonical JSON of the complete definition",
  "stage_start": 2,
  "stage_end": 4,
  "lanes": [
    {"lane_id": 0, "data_name": "arbitrary/name", "sample_name": "sample",
     "input_path": "/trusted/local/temporary-directory/0.pkl"}
  ]
}
```

内部 ID は実行全体で 0, 1, ... と割り当て、子の担当サブセットでも再採番しない。
任意の data_name は TAR パスに使わない。対応表は request.json のみに入れる。
params.pkl / synthesized.pkl は子単位で一度、value.pkl は担当レーンの分のみ送る。
None でもエントリを省略しない。入力 DataFrame 実体は TAR に含めない。

子 → 親（TAR/stdout）:

```text
lanes/0/images/<image_name>
lanes/0/value.pkl
lanes/1/images/<image_name>
lanes/1/value.pkl
error.json                  # 失敗時のみ
```

画像は既存 `_TarStreamingOutput` と同じく一枚単位のバッファから追加する。
画像内部のチャンク化は対象外。レーンの最後の map の戻り値だけを一度 pickle 化し送る。
アーカイブ全体はメモリにためず、w| / r|* で流す。stderr も別途 drain し、
大量のログで stdout の回収を止めない。応答のバージョンは要求 v1 に従い、error.json にも明示する。

```json
{
  "protocol_version": 1,
  "lane_id": 0,
  "data_name": "arbitrary/name",
  "stage_index": 3,
  "error_type": "ValueError",
  "message": "diagnostic message",
  "traceback": "Traceback ..."
}
```

実行前のエラーでは lane_id / data_name / stage_index を null にする。失敗した map 以降を
実行しない。親は error.json、非ゼロ終了、TAR 破損、未受信 / 重複 value.pkl を失敗とする。
stderr と終了コードも保持する。既存 API の CSV / Series と非構造化 `error` エントリは
この形式とは独立であり、移行期間中の既存動作を維持する。

## 実験と検証

```bash
uv run ruff check
uv run pytest tests/test_pipeline_transfer_probe.py -v
uv run pytest
```

実験は一子に複数レーンを割り当て、全 Manifest を再構築する。入力 NamedTuple 型と
pickle 対象 dataclass は import 可能なモジュールに置く（-m 起動時も同じ型を import）。
レーン分割は scanner の代わりに小さな lane 列の fixture を使う。

| 検証対象 | 自動テストでの観測 |
| --- | --- |
| preprocess と入力 | 元 Inputs → 複数 DataFrame を持つ Prepared に変換。前処理は一回、元入力不変、型を保持してファイル復元 |
| 任意値 | DataFrame / dataclass / None / dict・tuple・bytes・set をファイルと要求・応答で往復 |
| 連続 map | lambda を含む pickle 不可の中間 value を直接受け渡し、最終 Feature のみ送信。一子で二レーン |
| 分離 | 各 map で入力二枚と params / synthesized を変更しても次 map・別レーンへ伝播しない。value の変更は戻り値経由で保持 |
| 親 | 全入力復元、入力と values の外側 Mapping 保護、params の独立コピー、values 要素と synthesized の参照維持 |
| Notebook | pickle を失敗させる monkeypatch と lambda を含む object 列でも DataFrame コピーが動作 |
| 逐次受信 | 画像二枚→レーン値の順に回収。親が値を更新して ack ファイルを作るまで子が待機し、終了前受信を確認 |
| 実行前拒否 | Manifest・protocol_version 不一致、空 / 範囲外 / 非整数 / 非 map を含む範囲。イベントなし、状態不変 |
| 失敗 | pickle 不可、欠損 / 重複結果、TAR 切断・破損、非ゼロ終了、error.json を拒否。絶対 stage_index・レーン・traceback を保持 |
| コミットとバリア | レーン0の画像・値を保持したままレーン1が失敗し、後続処理へ進まない |
| 寿命 | 成功・失敗とも子の回収後にファイル削除。入力保存失敗は子起動前に一時領域削除 |

2026-10-20、macOS / Python 3.12.12 / pytest 9.1.1 で検証した。
`uv run ruff check` は `All checks passed!`、`uv run pytest` は
**138 passed in 26.87s**（既存100件、本実験38件）。追加 Python ファイルの Ruff 整形も実施した。

具体的には二レーンの元 score=1,2 を preprocess で11,12にし、同じ子の二つの map で
Feature(score=13), Feature(score=14) を受信した。各レーンで画像二枚→value の順に届き、
子が生存中に親 values を更新して ack できた。110,000 bytes の stderr を同時に drain した。
レーン1の入力復元を失敗させたケースでは、レーン0の Feature(13) と画像二枚を保持したまま
エラーになり、後続処理には到達しなかった。一時ファイルは子終了の確認後に削除された。

### 制限と後続への引き継ぎ

- 実画像のレンダリング・Output 製品統合は既存 API の範囲。今回の実験は画像に相当する
  bytes を一枚ずつ流して順序・到着を検証する。
- TAR の標準 record バッファでは小さい画像・結果の送信が子終了まで遅れる。
  実験はエントリごとに書込バッファを排出し、親の読込 bufsize を512にする。
  書込側は tarfile の内部ストリームに触る最小実験であり、#85 では適切な flush の
  抽象化とレイテンシを検証する必要がある。アーカイブ全体のバッファリングは不要だった。
- 一子の失敗処理と回収のみ検証。多数の子の同時実行、他子の停止・回収、共有ディスクの
  負荷、Windows、異なる Python バージョン、大容量の速度・メモリ測定は未検証。
  ハーネスの30秒 watchdog はテストのハング防止であり、製品の timeout 方針ではない。
- #83: Notebook を含む正式な逐次 Executor と DataFrame 戻り値の検証。
- #84: Map Segment の実並列実行・スケジューリング・全子の停止と回収・バリア。
- #85: Output、構造化例外、最終ストリーミングの製品実装。
- #86: 公開ビルダー・型・ExecutionContext。#87: scanner 統合・旧 API の移行と削除。
