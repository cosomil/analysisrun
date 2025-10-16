import inspect
from io import BytesIO
from os import getcwd
from pathlib import Path
from sys import stdin
from typing import Any, Optional, Type, TypeVar, get_origin

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import deprecated

from .__pipe import read_tar_as_dict

T = TypeVar("T", bound=BaseModel)


def _prompt_for_value(
    field_type: Type, field_name: str, description: Optional[str], field_info=None
) -> Any:
    """
    フィールドの型に応じて適切な入力方法でユーザーから値を取得します。

    Parameters
    ----------
    field_type
        フィールドの型
    field_name
        フィールド名
    description
        フィールドの説明
    field_info
        フィールドの追加情報（デフォルト値など）
    """
    origin = get_origin(field_type)

    # デフォルト値の取得と判定
    default_value = (
        getattr(field_info, "default", PydanticUndefined)
        if field_info
        else PydanticUndefined
    )
    has_default = default_value is not PydanticUndefined

    # デフォルト値の表示用文字列
    default_str = f" [デフォルト: {default_value}]" if has_default else ""
    prompt = (
        f"{description} ({field_name}){default_str}: "
        if description
        else f"{field_name}{default_str}: "
    )

    # NamedTuple型の場合
    if (
        inspect.isclass(field_type)
        and issubclass(field_type, tuple)
        and hasattr(field_type, "_fields")
    ):
        print(prompt)
        named_tuple_values = {}

        for sub_field in field_type._fields:  # type: ignore
            sub_value = input(f"  - {sub_field}: ")
            # 入力がない場合のフラグ
            if not sub_value and has_default:
                return default_value
            named_tuple_values[sub_field] = sub_value

        return field_type(**named_tuple_values)

    # 通常のタプル型の場合
    elif origin is tuple or field_type is tuple:
        print(prompt)
        tuple_values = []

        # タプルの要素数がわかる場合
        args = getattr(field_type, "__args__", None)
        if args:
            for i, arg_type in enumerate(args):
                sub_value = input(f"  - {i}: ")
                # 空入力があり、デフォルト値が存在する場合は即時適用
                if not sub_value and has_default:
                    return default_value
                tuple_values.append(sub_value)
        else:
            # 要素数が不明の場合
            first_element = True
            while True:
                sub_value = input(f"  - {len(tuple_values)} (空白で終了): ")
                # 最初の入力が空で、デフォルト値がある場合はデフォルト値を適用
                if not sub_value and first_element and has_default:
                    return default_value

                # 要素追加後に空入力なら入力終了
                if not sub_value and len(tuple_values) > 0:
                    break

                if sub_value:  # 空でない入力の場合
                    tuple_values.append(sub_value)
                    first_element = False
                else:
                    # 初回の空入力でデフォルト値がない場合は空のタプルを返す
                    break

        return tuple(tuple_values)

    # その他の型の場合
    else:
        value = input(prompt)

        # 入力がない場合にデフォルト値が設定されている場合
        if not value and has_default:
            return default_value

        return value


def scan_model_input(model_class: Type[T]) -> T:
    """
    モデルのフィールドをインタラクティブに入力し、モデルのインスタンスを返します。
    標準入力がリダイレクトされている場合はtar形式としてデータを読み込み、ファイル名／データを名前／データに変換します。

    note: フィールドがさらにBaseModelを継承している場合の処理は未実装です。

    Parameters
    ----------
    model_class
        Pydanticモデルクラス
    """

    if not stdin.isatty():
        _in = read_tar_as_dict(stdin.buffer)
        return model_class(**_in)

    # 有効な入力値を保存する辞書
    valid_inputs = {}
    field_values = {}

    while True:
        try:
            # モデルのフィールドを取得
            fields = model_class.__annotations__
            field_values = {}

            # 前回有効だった入力を引き継ぎ
            field_values.update(valid_inputs)

            # エラーフィールドのセット（初回は全フィールドを入力対象とする）
            error_fields = set(fields.keys()) - set(valid_inputs.keys())

            if not error_fields:
                # すべてのフィールドが有効な場合は全フィールドを入力対象にする
                error_fields = set(fields.keys())

            # 各フィールドについて入力を促す
            for field_name, field_type in fields.items():
                # エラーのあったフィールドのみ再入力を要求
                if field_name not in error_fields:
                    continue

                field_info = model_class.model_fields.get(field_name)
                description = (
                    field_info.description
                    if field_info and field_info.description
                    else None
                )

                # 値の入力
                field_values[field_name] = _prompt_for_value(
                    field_type, field_name, description, field_info
                )

            # モデルのインスタンスを作成してみる
            return model_class(**field_values)

        except Exception as e:
            print("\n⚠️ 入力値にエラーがあります:")

            # Pydanticのバリデーションエラーの場合、エラーフィールドを特定
            error_fields = set()

            if isinstance(e, ValidationError):
                # エラーメッセージをフィールドごとにグループ化
                field_errors = {}
                for error in e.errors():
                    if error["loc"]:
                        field_name = error["loc"][0]
                        error_fields.add(field_name)
                        if field_name not in field_errors:
                            field_errors[field_name] = []
                        field_errors[field_name].append(error["msg"])

                # エラー内容を表示
                for field_name, errors in field_errors.items():
                    print(f"  {field_name}:")
                    for error_msg in errors:
                        print(f"    - {error_msg}")

                # エラーがないフィールドの値を保存
                for field_name, value in field_values.items():
                    if field_name not in error_fields:
                        valid_inputs[field_name] = value

                print("エラーとなった項目を再入力してください")
            else:
                # その他の例外の場合は全フィールドを再入力
                print(f"エラーが発生しました: {str(e)}")
                print("すべての項目を再入力してください")
                valid_inputs = {}  # 有効な入力をリセット

            print()


@deprecated("削除予定。VirtualFileを使用してください。")
class FilePath(str):
    """
    ファイルパスを表す文字列型。バリデーションの際にファイルの存在を確認します。
    文字列の前後にシングルクォートがある場合は削除します。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if not isinstance(v, str):
                return v
            if v.startswith("'") and v.endswith("'"):
                v = v[1:-1]
            p = Path(v)
            if not p.exists():
                raise ValueError(f"ファイルが存在しません: '{v}'")
            if not p.is_file():
                raise ValueError(f"ファイルのパスを指定してください: '{v}'")
            return v

        return core_schema.no_info_plain_validator_function(validate)


@deprecated("削除予定")
class DirectoryPath(str):
    """
    ディレクトリパスを表す文字列型。バリデーションの際にディレクトリの存在を確認します。
    文字列の前後にシングルクォートがある場合は削除します。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if not isinstance(v, str):
                return v
            if v.startswith("'") and v.endswith("'"):
                v = v[1:-1]
            p = Path(v)
            if not p.exists():
                raise ValueError(f"ディレクトリが存在しません: '{v}'")
            if not p.is_dir():
                raise ValueError(
                    f"ディレクトリ（フォルダー）のパスを指定してください: '{v}'"
                )
            return v

        return core_schema.no_info_plain_validator_function(validate)


class VirtualFile(Path):
    """
    仮想ファイルを扱う型。Pathオブジェクトまたはio.BytesIOオブジェクトを受け入れます。
    Pathオブジェクトのようにふるまうほか、io.BytesIOオブジェクトが与えられた場合にはファイルのように振る舞います。
    pandasのread_csvで読み込みができることを確認しています。

    Pathあるいは文字列が与えられた場合:
    - ファイルの存在を確認し、存在しなければエラーとなります。

    io.BytesIOが与えられた場合:
    - 作業ディレクトリに存在する可能ファイルのように振る舞います。
    - ファイル名は仮想的に"virtual-file"とします。
    """

    def __init__(self, v: Path | BytesIO):
        if isinstance(v, Path):
            super().__init__(v)
        elif isinstance(v, BytesIO):
            super().__init__(Path(getcwd()) / "virtual-file")

            # io.BytesIOが与えられた場合のみ、FileLikeオブジェクトとして振る舞うためのメソッドを追加
            def read(self: VirtualFile | None = None) -> bytes:
                return v.read()

            def iter(self: VirtualFile | None = None):
                raise NotImplementedError()

            self.read = read
            self.__iter__ = iter

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if isinstance(v, (str, Path)):
                if isinstance(v, str):
                    if v.startswith("'") and v.endswith("'"):
                        v = v[1:-1]
                    v = Path(v)
                if not v.exists():
                    raise ValueError(f"ファイルが存在しません: '{v}'")
                if not v.is_file():
                    raise ValueError(f"ファイルのパスを指定してください: '{v}'")
                return cls(v)
            elif isinstance(v, BytesIO):
                return cls(v)  # type: ignore
            raise TypeError(f"Unsupported type: {type(v)}")

        return core_schema.no_info_plain_validator_function(validate)
