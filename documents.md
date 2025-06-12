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
show=Trueの場合、保存後にNotebookへの表示を実行する。

## `class AnalyzeArgs`

**メンバ変数**

- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト。
- `fields: Fields`: 対象となるレーンのデータを視野ごとに探索するためのスキャナー。
- `fields_for_enhancement: list[Fields]`: 各データを別の観点から解析し、補強するためのスキャナーのリスト。
- `output: Output`: 画像を保存するためのOutput実装。

## `class PostprocessArgs`

**メンバ変数**

- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト。
- `analysis_results: pd.DataFrame`: 解析結果を格納したDataFrame。

## `class NotebookRunner`

主にJupyter notebookでの使用を想定したrunner。

### `NotebookRunner.run(ctx: Context, target_data: list[str], whole_data: CleansedData, data_for_enhancement: list[CleansedData], field_numbers: Optional[list[int]], output: Optional[Output]) -> pd.DataFrame`

各レーンごとに数値解析を実行し、解析結果を結合したDataFrameを返す

**引数**
- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト
- `target_data: list[str]`: 対象データのリスト
- `whole_data: CleansedData`: クレンジングされた解析対象データ
- `data_for_enhancement: list[CleansedData]`: 各データを別の観点から解析し、補強するためのデータのリスト
- `field_numbers: Optional[list[int]]`: スキャン対象となる視野番号のリスト
- `output: Optional[Output]`: 画像を保存するためのOutput実装

## `class ParallelRunner`

マルチプロセスで並列処理するrunner。

### `ParallelRunner.run(ctx: Context, target_data: list[str], whole_data: CleansedData, data_for_enhancement: list[CleansedData], field_numbers: Optional[list[int]], output: Optional[Output]) -> pd.DataFrame`

各レーンごとに数値解析を実行し、解析結果を結合したDataFrameを返す

**引数**
- `ctx: Context`: 解析全体に関わる情報を格納するコンテキストオブジェクト
- `target_data: list[str]`: 対象データのリスト
- `whole_data: CleansedData`: クレンジングされた解析対象データ
- `data_for_enhancement: list[CleansedData]`: 各データを別の観点から解析し、補強するためのデータのリスト
- `field_numbers: Optional[list[int]]`: スキャン対象となる視野番号のリスト
- `output: Optional[Output]`: 画像を保存するためのOutput実装

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

## `class Fields`

レーンのデータを視野ごとにスキャンする

### `Fields.skip_empty_fields()`

データのない視野をスキップするスキャナーを作成する

## `class Lanes`

データ全体をレーンごとにスキャンする

---

# analysisrun.cleansing

数値解析対象として意図されていないデータの混入を防ぐためのクレンジング処理と、クレンジング済みであることを表すデータ型を提供します。

## `class CleansedData`

データクレンジング処理後の解析対象データ

### `filter_by_entity(data: pd.DataFrame | CleansedData, entity: str) -> CleansedData`

指定されたエンティティ名でデータをフィルタリングする。

**引数**
- `data: pd.DataFrame | CleansedData`: 解析対象データ
- `entity: str`: 数値解析の対象となるエンティティ名(Entity列)

---

# analysisrun.interactive

## `class FilePath(str)`

ファイルパスを表す文字列型。バリデーションの際にファイルの存在を確認します。
文字列の前後にシングルクォートがある場合は削除します。

## `class DirectoryPath(str)`

ディレクトリパスを表す文字列型。バリデーションの際にディレクトリの存在を確認します。
文字列の前後にシングルクォートがある場合は削除します。

### `scan_model_input(model_class: Type[T]) -> T`

モデルのフィールドをインタラクティブに入力し、モデルのインスタンスを返します。

note: フィールドがさらにBaseModelを継承している場合の処理は未実装です。

**引数**
- `model_class: Type[T]`: Pydanticモデルクラス

---

