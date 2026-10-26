# streaming モードの入出力仕様

## 概要

`streaming` モードは、解析プログラムに **stdin から TAR アーカイブ**を渡し、**stdout から gzip 圧縮 TAR アーカイブ**を受け取る非対話インターフェースです。複数レーンを解析し、レーン別の画像と最終的な集計結果を返します。

- **stdin**: 非圧縮 TAR または gzip 圧縮 TAR（`tar.gz`）
- **stdout**: gzip 圧縮 TAR（`tar.gz`）。標準出力はテキストログではなくバイナリプロトコルです。
- **stderr**: 診断情報・例外 traceback など。機械的な成否判定は、終了コードおよび stdout の `error` エントリで行います。
- **終了コード**: 正常終了は `0`、入力・利用方法のエラーは `2`、解析実行中のエラーは `1`。

通常、Jupyter Notebook ではない環境で stdin が TTY でなければ `streaming` が選択されます。`ANALYSISRUN_INTERACTIVITY=terminal` を設定すると対話的実行になります。`ANALYSISRUN_MODE` は通常のモード強制指定には利用できず、特別な値 `showschema` を指定すると入力仕様の JSON を出力して終了します。`manual_input` が与えられていても、`streaming` では stdin の TAR 入力を使います。

**重要:** TAR が読み込めたことや、画像が受信できたことだけでは解析成功を意味しません。受信側はストリーム終端・プロセス終了コード・`error` の有無を確認し、成功を確定してください。

## 1. 入力仕様

### TAR エントリ

| エントリ名 | データ形式 | 必須 | 内容 |
| --- | --- | --- | --- |
| `params` | UTF-8 JSON | 必須 | 解析プログラムが定義する Pydantic `Params` モデルに従うパラメータ |
| `sample_names` | CSV | 必須 | レーン識別子とサンプル名の対応 |
| `image_analysis_results/{name}` | CSV | 定義された各 `name` について必須 | 解析プログラムが定義する画像解析結果データセット |

`{name}` は解析プログラムの `ImageAnalysisResults` のフィールド名です。例えば `measurements` を定義したプログラムでは `image_analysis_results/measurements` が必要です。任意の名前で置き換えられるわけではありません。

**CSV エントリには PAX ヘッダー `is_file=true` を設定します。** このヘッダーのあるエントリはファイルとして受け取られます。ヘッダーのないエントリは UTF-8 テキストとして扱われるため、CSV の入力に適しません。`params` は通常の文字列エントリとして格納し、JSON としてデコードされます。

入力には TAR のファイルエントリを使ってください。ディレクトリエントリを明示的に追加する必要はありません。エントリ名は `/` 区切りで指定します。

### `params`

たとえば解析プログラムが次を定義している場合:

```python
from pydantic import BaseModel

class Params(BaseModel):
    minimum_score: float = 0.8
```

エントリ `params` の内容は次の JSON になります。

```json
{"minimum_score": 0.8}
```

必須フィールド、型、既定値などは解析プログラムによって異なります。`showschema` で確認できます。

### `sample_names`

`sample_names` は少なくとも次の列を持つ CSV です。

```csv
data,sample
0000,QC
0001,Sample A
```

- `data`: レーンの識別子。解析結果のファイル名等と対応します。
- `sample`: 利用者向けのサンプル名。

空の対応表は受け付けられません。`data` は先頭のゼロを含む識別子として扱うため、送信側でも文字列として保持してください。

### `image_analysis_results/{name}`

各エントリは解析対象の CSV です。必要な列やクレンジング処理は解析プログラムに依存します。例えば QC フィードバックの例では `image_analysis_results/measurements` に `MultiPointIndex,Entity,Value,Filename` を含む CSV を渡します。

入力読込時にライブラリは CSV を読み込み、解析プログラムが設定したクレンジングを適用し、ファイル名からレーン情報を正規化します。`Filename` と `sample_names` の `data` の対応が正しいことを確認してください。CSV の具体的な列要件は対象の解析プログラムのドキュメントも参照してください。

### 動的スキーマの取得（`showschema`）

解析プログラムに次の環境変数を設定して起動すると、解析を開始せずに入力仕様の JSON を stdout に出力して正常終了します。

```bash
ANALYSISRUN_MODE=showschema uv run python -m example.qc-feedback > schema.json
```

出力される JSON は概ね次の形です（フィールドの schema とエントリ数は解析プログラムに依存します）。

```json
{
  "schema_version": "1",
  "transport": {
    "type": "tar",
    "compression": ["tar", "tar.gz"],
    "path_separator": "/"
  },
  "tar_entries": [
    {
      "path": "params",
      "required": true,
      "content_type": "application/json",
      "description": "解析全体に関わるパラメータ",
      "pax_headers": {},
      "json_schema": {}
    },
    {
      "path": "sample_names",
      "required": true,
      "content_type": "text/csv",
      "description": "サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
      "pax_headers": {"is_file": "true"}
    },
    {
      "path": "image_analysis_results/measurements",
      "required": true,
      "content_type": "text/csv",
      "description": "解析対象の測定値",
      "pax_headers": {"is_file": "true"}
    }
  ]
}
```

`params` の `json_schema` には実際には Pydantic の JSON Schema が入ります。上記の空オブジェクトは説明用の省略です。`showschema` の stdout は **JSON テキスト**であり、通常の streaming 応答 TAR ではありません。

## 2. 出力仕様

stdout は gzip 圧縮 TAR です。成功時には、解析が生成した画像と最終的な DataFrame に対応する結果エントリが含まれます。

| エントリ名 | 内容 |
| --- | --- |
| `{data}/images/{image_name}` | 解析処理が生成した画像のバイナリ |
| `result_csv` | 最終 DataFrame 全体の CSV |
| `{data}/result_json` | 最終 DataFrame の各行を JSON オブジェクトにしたもの |

`{data}` はレーンの `data_name`（画像）または最終結果 DataFrame の `data` 列の値（JSON）です。`data` 列がない場合、`result_json` のパスには DataFrame の行インデックスが用いられます。`image_name` は解析プログラムが画像出力時に指定した名前です。

### データ表現

- `result_csv` は UTF-8 CSV で、ヘッダー行を含み、DataFrame のインデックス列は出力しません。
- `{data}/result_json` は UTF-8 JSON オブジェクトです。
- `{data}/result_json` では、元の DataFrame で欠損している値（`None`、`numpy.nan`、`pandas.NA`、`pandas.NaT`）は JSON の **`null`** として出力します。元の値が欠損でない場合は従来どおり文字列として出力します。たとえば `123` は `"123"`、`True` は `"True"` になります。通常の文字列 `"None"` や `"nan"` は `null` に変換しません。
- 出力 TAR の画像・`result_csv`・`result_json` エントリには PAX ヘッダー `is_file=true` が付与されます。`error` エントリには付与されません。画像エントリの `image_type` は画像出力で指定された値が設定され、内部的に `None` が渡された場合はヘッダーを省略します。
- 画像のデータ形式は解析プログラムの出力指定に依存し、必ず PNG とは限りません。

### 順序とストリーミング

画像は解析中に出力され、受信側は解析完了前から受信できます。一方、`result_csv` とレーン別の `result_json` は最終的な DataFrame の生成後に出力されます。現行実装では `result_csv` の後に `result_json` が続きます。

並列 map の実行順序・画像の到着順序には依存しないでください。同一レーンの画像についても、外部の受信プログラムは単に TAR エントリ名で識別する設計を推奨します。

**TAR 内のエントリ名は重複できません。** 同じレーンで同名画像を複数回生成すると、ステージが異なる場合もエラーになります。別レーンで同じ画像名を使うことは、`{data}` が異なる限り可能です。最終結果の `data` が重複し、`{data}/result_json` が衝突する場合もエラーになります。

## 3. 失敗時の仕様

### エラー応答

解析が始まった後に例外が発生した場合、可能な限り stdout の gzip TAR を正常終端し、最後の**データエントリ**に `error` を出力して、終了コード `1` で終了します。

- エントリ名: `error`
- 内容: UTF-8 のエラーメッセージ本文（JSON オブジェクトではありません）
- それまでに送信した画像は取り消されず、部分的な出力が残り得ます。
- 成功時の最終結果（`result_csv`、`result_json`）がすべて揃うとは限りません。
- 出力先の切断など、書き込みが継続できない障害では `error` や正常な TAR 終端は保証されません。

入力 TAR の読込・モデル検証・クレンジングなど、実行開始**前**に失敗した場合は終了コード `2` となります。この段階では別のエラーハンドラからエラー専用の gzip TAR が出力され、`error` エントリに UTF-8 メッセージが格納されます。TAR 自体を復元できない障害もあるため、`error` が必ず取得できることを前提にしないでください。

### 受信側の成功判定

次の条件をすべて満たすときだけ成功として扱ってください。

1. プロセスが終了コード `0` で終了した。
2. stdout の gzip と TAR を最後まで読み取り、破損なく復元できた。
3. TAR に `error` エントリがない。
4. 利用者が必要とする `result_csv` 等の結果エントリが存在し、解析プログラムが期待する内容になっている。

画像や一部の結果が既に受信されていても、エラーが発生した実行は失敗として扱います。stderr は調査用に記録できますが、文字列パターンによる成否判定には使用しないでください。

## 4. 実行例

以下は `example/qc-feedback` を使った例です。リポジトリのルートで実行し、サンプル付属の `input/measurements.csv`、`input/samples.csv` を使います。

### 入力 TAR の生成

```python
# create_request.py
import io
import json
import tarfile
from pathlib import Path

root = Path("example/qc-feedback/input")

with tarfile.open("request.tar.gz", "w:gz", format=tarfile.PAX_FORMAT) as archive:
    def add(name: str, data: bytes, *, csv: bool = False) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        if csv:
            info.pax_headers = {"is_file": "true"}
        archive.addfile(info, io.BytesIO(data))

    add("params", json.dumps({"minimum_score": 0.8}).encode("utf-8"))
    add("sample_names", (root / "samples.csv").read_bytes(), csv=True)
    add(
        "image_analysis_results/measurements",
        (root / "measurements.csv").read_bytes(),
        csv=True,
    )
```

```bash
uv run python create_request.py
uv run python -m example.qc-feedback < request.tar.gz > response.tar.gz
echo $?
```

正常に完了すると、`response.tar.gz` には次のようなエントリが入ります（レーン数と画像名は入力と解析実装によって変わります）。

```text
0000/images/0000_qc.png
0001/images/0001_qc.png
...
result_csv
0000/result_json
0001/result_json
...
```

### レスポンスの読取

```python
# inspect_response.py
import json
import tarfile

with tarfile.open("response.tar.gz", "r:gz") as archive:
    entries = {
        member.name: archive.extractfile(member).read()
        for member in archive
        if member.isfile()
    }

if "error" in entries:
    raise RuntimeError(entries["error"].decode("utf-8"))

print(entries["result_csv"].decode("utf-8"))
for name, data in entries.items():
    if name.endswith("/result_json"):
        print(name, json.loads(data))
```

上記は小規模な応答の内容を確認するための例です。このコードだけでは gzip の CRC・footer や TAR 終端ブロックの完全性を厳密に検証できないため、実運用では別途検証してください。実運用で大きな画像を扱う場合は全エントリをメモリに保持せず、`tarfile.open(..., mode="r|gz")` 等で逐次読み出してください。また、**このスクリプトとは別に、起動した解析プロセスの終了コードを必ず確認**してください。

### エラー応答の例

必須エントリのない空の TAR を渡すと、入力エラーを再現できます。

```bash
uv run python -c 'import tarfile; tarfile.open("invalid-request.tar.gz", "w:gz").close()'
uv run python -m example.qc-feedback < invalid-request.tar.gz > error-response.tar.gz
echo $?  # 2
```

`error-response.tar.gz` は `error` のみを含み、その内容は次の UTF-8 メッセージです。
原因の traceback は stderr に出力されます。

```text
入力データの読み込みに失敗しました。入力形式を確認してください。
```

解析開始後のエラーは、[クライアント例](../example/streaming_client.py)でも再現できます。

```bash
uv run python example/streaming_client.py --fail
```

この例は QC サンプルへ測定値 0 を渡します。解析プロセスは終了コード `1` で終了し、
`error` のみの gzip TAR を返します。メッセージは `reference must be positive` です。
クライアントは終了コードと `error` を検査し、自身も終了コード `1` で終了します。
同じクライアントを `--fail` なしで起動すると、正常系の入出力も確認できます。
このクライアントには gzip の CRC・footer と TAR 終端ブロックの検証例も含まれます。

## 5. 運用上の注意

- stdout には gzip TAR が流れるため、解析処理のログを stdout に直接書き込まないでください。ライブラリは解析処理中の通常の `print()` を stderr にリダイレクトしますが、プロセス全体・ネイティブ拡張・サブプロセスによる直接書き込みまで保証するものではありません。
- 画像は逐次出力されますが、最終結果は解析完了後に確定します。**受信中に見つかった画像を「成功済み」として公開しない**設計を推奨します。

## 関連資料

- [README](../README.md)
- [Pipeline API](pipeline.md)
- [QC フィードバック例](../example/qc-feedback/README.md)
- [入力スキーマの実装](../src/analysisrun/input.py)
- [外部入出力の実装](../src/analysisrun/context.py)
- [Streaming エントリポイント](../src/analysisrun/pipeline_entrypoint.py)
- [出力エントリの実装](../src/analysisrun/pipeline_worker.py)
- [外部 TAR の読み取り](../src/analysisrun/tar.py)
