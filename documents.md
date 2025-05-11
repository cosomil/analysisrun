# analysisrun.runner

## `class Output(Protocol)`

matplotlib.figure.Figureを保存する。

**引数**
- `fig`: 保存するFigure。
- `name`: 保存するファイル名。
- `image_type`: 画像タイプ。 実際の画像保存処理のヒントとなります。
- `kwargs`: savefigに渡すキーワード引数。

## `class DefaultOutput`

matplotlib.figure.Figureを保存する。
shoe=Trueの場合、保存後にNotebookへの表示を実行する。

## `class AnalyzeArgs`

**メンバ変数**

- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト。
    dataclassを使用するのが望ましい。
- `lane: scanner.LaneDataScanner`: 対象となるレーンのデータを探索するためのスキャナー。
- `output: Output`: 画像を保存するためのOutput実装。

## `class NotebookRunner`

主にJupyter notebookでの使用を想定したrunner。

### `NotebookRunner.run(ctx: Context, analyze: Callable[[AnalyzeArgs[Context]], pd.Series]) -> pd.DataFrame`

各レーンごとに画像解析を実行する。
レーンごとの解析結果を結合したDataFrameを返す。

**引数**
- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト。 dataclassを使用するのが望ましい。
- `analyze: Callable[[AnalyzeArgs[Context]], pd.Series]`: 解析関数。 解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。

## `class ParallelRunner`

マルチプロセスで並列処理するrunner。

### `ParallelRunner.run(ctx: Context, analyze: Callable[[AnalyzeArgs[Context]], pd.Series]) -> pd.DataFrame`

各レーンごとに画像解析を実行する。
レーンごとの解析結果を結合したDataFrameを返す。

**引数**
- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト。 dataclassを使用するのが望ましい。
- `analyze: Callable[[AnalyzeArgs[Context]], pd.Series]`: 解析関数。 解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。

---

# analysisrun.helper

### `read_dict(filename: str, key: str, value: str) -> Dict[str, str]`

CSVファイルを読み込み、指定したカラムをキーと値にして辞書を作成する。

**引数**
- `filename: str`: 読み込むCSVファイルのパス
- `key: str`: 辞書のキーとなるカラム名
- `value: str`: 辞書の値となるカラム名

### `is_float(x: float | None) -> TypeGuard[float]`

---

# analysisrun.scanner

## `class LaneDataScanner`

レーンのデータを視野ごとにスキャンする

### `LaneDataScanner.each_viewpoint(filter: Optional[Filter], skip_empty_viewpoints: bool) -> Generator[pd.DataFrame, Any, None]`

視野ごとのデータを抽出するジェネレータ

**引数**
- `filter: Optional[Filter]`: フィルタ条件
- `skip_empty_viewpoints: bool`: データのない視野をスキップするかどうか

## `class Scanner`

データ全体をレーンごとにスキャンする

### `Scanner.each_lane(filter: Optional[Filter])`

各レーンのデータを読み込むLaneDataScannerを生成するジェネレータ

**引数**
- `filter: Optional[Filter]`: フィルタ条件

---

# analysisrun.interactive

### `scan_model_input(model_class: Type[T]) -> T`

モデルのフィールドをインタラクティブに入力し、モデルのインスタンスを返します。

note: フィールドがさらにBaseModelを継承している場合の処理は未実装です。

**引数**
- `model_class: Type[T]`: Pydanticモデルクラス

---

