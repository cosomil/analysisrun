import inspect
from dataclasses import is_dataclass
from io import BytesIO
from os import getcwd
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Type,
    TypeGuard,
    TypeVar,
    get_origin,
)

import questionary
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import deprecated

T = TypeVar("T")


def _get_field_descriptions(typ) -> str | None:
    if type(typ) is type(Annotated[None, None]):  # 'typing._AnnotatedAlias'クラス
        for _, x in enumerate(typ.__metadata__):  # type: ignore
            return _get_field_descriptions(x)
    if type(typ) is type(Field()):
        return typ.description  # type: ignore
    return None


def _get_default_value(field_info) -> Any:
    if field_info is None or field_info.is_required():
        return PydanticUndefined
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        return field_info.default_factory()
    return PydanticUndefined


def _format_validation_error(e: ValidationError, field_name: str) -> str:
    messages = []
    for error in e.errors():
        loc = list(error.get("loc", ()))
        if loc and loc[0] == field_name:
            loc = loc[1:]
        suffix = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc
        )
        messages.append(f"{field_name}{suffix}: {error['msg']}")
    return "\n".join(messages) if messages else str(e)


def _validate_and_store_field_value(
    model_class: Type[BaseModel], draft_model: BaseModel, field_name: str, value: Any
) -> Any:
    try:
        model_class.__pydantic_validator__.validate_assignment(
            draft_model, field_name, value
        )
    except AttributeError:
        # 相関チェックが未入力フィールドに依存する場合は、最終的なモデル検証に委ねる
        field_info = model_class.model_fields[field_name]
        annotation = field_info.annotation
        if field_info.metadata:
            annotation = Annotated[annotation, *field_info.metadata]
        parsed_value = TypeAdapter(annotation).validate_python(value)
        setattr(draft_model, field_name, parsed_value)
    return getattr(draft_model, field_name)


def _ask_text(
    prompt: str,
    model_class: Type[BaseModel],
    draft_model: BaseModel,
    field_name: str,
    default_value: Any,
    input_method: Literal["text", "path"] = "text",
) -> Any:
    has_default = default_value is not PydanticUndefined

    def validate(text: str) -> bool | str:
        candidate = default_value if text == "" and has_default else text
        try:
            _validate_and_store_field_value(
                model_class, draft_model, field_name, candidate
            )
        except ValidationError as e:
            return _format_validation_error(e, field_name)
        return True

    q: questionary.Question
    match input_method:
        case "text":
            q = questionary.text(prompt, validate=validate)
        case "path":
            q = questionary.path(prompt, validate=validate)

    answer = q.unsafe_ask()
    candidate = default_value if answer == "" and has_default else answer
    return _validate_and_store_field_value(
        model_class, draft_model, field_name, candidate
    )


def _ask_questionary_text(
    prompt: str, validate: Optional[Callable[[str], bool | str]] = None
) -> str:
    answer = questionary.text(
        prompt, validate=validate if validate is not None else (lambda _: True)
    ).ask()
    if answer is None:
        raise KeyboardInterrupt()
    return answer


def _prompt_for_value(
    model_class: Type[BaseModel],
    draft_model: BaseModel,
    field_type: Type,
    field_name: str,
    description: Optional[str],
    field_info=None,
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
    default_value = _get_default_value(field_info)
    has_default = default_value is not PydanticUndefined

    # デフォルト値の表示用文字列
    default_str = f" [デフォルト: {default_value}]" if has_default else ""
    prompt = (
        f"{description} ({field_name}){default_str}:"
        if description
        else f"{field_name}{default_str}:"
    )

    # NamedTuple型の場合
    if (
        inspect.isclass(field_type)
        and issubclass(field_type, tuple)
        and hasattr(field_type, "_fields")
    ):
        questionary.print(prompt, style="bold")
        named_tuple_values = {}
        field_defaults = field_type._field_defaults or dict()  # type: ignore
        assert isinstance(field_defaults, dict)

        for sub_field in field_type._fields:  # type: ignore
            default = field_defaults.get(sub_field)
            description = (
                _get_field_descriptions(default) if default is not None else None
            )
            prompt = f"{description} ({sub_field}):" if description else f"{sub_field}:"
            sub_value = _ask_questionary_text(prompt)
            # 入力がない場合のフラグ
            if not sub_value and has_default:
                return default_value
            named_tuple_values[sub_field] = sub_value

        return field_type(**named_tuple_values)

    # 通常のタプル型の場合
    elif origin is tuple or field_type is tuple:
        questionary.print(prompt, style="bold")
        tuple_values = []

        # タプルの要素数がわかる場合
        args = getattr(field_type, "__args__", None)
        if args:
            for i, arg_type in enumerate(args):
                sub_value = _ask_questionary_text(f"{i}:")
                # 空入力があり、デフォルト値が存在する場合は即時適用
                if not sub_value and has_default:
                    return default_value
                tuple_values.append(sub_value)
        else:
            # 要素数が不明の場合
            first_element = True
            while True:
                sub_value = _ask_questionary_text(f"{len(tuple_values)} (空白で終了):")
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

    # Pydanticモデルの場合（ネストした入力）
    elif _is_pydantic_model(field_type):
        return _scan_object(field_type, field_name)

    # その他の型の場合（custom_inputを含む）
    else:
        input_method = _get_custom_input_method(field_type)
        return _ask_text(
            prompt=prompt,
            model_class=model_class,
            draft_model=draft_model,
            field_name=field_name.split(".")[-1],
            default_value=default_value,
            input_method=input_method,
        )


def _scan_object(model_class: Type[T], parent: Optional[str]) -> T:
    assert issubclass(model_class, BaseModel), (
        "model_class must be a Pydantic BaseModel"
    )

    fields = model_class.model_fields
    valid_inputs = {}

    for field_name, field_type in fields.items():
        if field_type.annotation is type(None):
            valid_inputs[field_name] = None

    while True:
        field_values = {}
        try:
            field_values.update(valid_inputs)

            draft_model = model_class.model_construct()
            for field_name, value in valid_inputs.items():
                setattr(draft_model, field_name, value)

            error_fields = set(fields.keys()) - set(valid_inputs.keys())
            if not error_fields:
                error_fields = set(fields.keys())

            for field_name, field_type in fields.items():
                if field_name not in error_fields:
                    continue

                field_info = model_class.model_fields.get(field_name)
                description = (
                    field_info.description
                    if field_info and field_info.description
                    else None
                )

                assert field_type.annotation is not None
                display_field_name = f"{parent}.{field_name}" if parent else field_name
                value = _prompt_for_value(
                    model_class=model_class,
                    draft_model=draft_model,
                    field_type=field_type.annotation,
                    field_name=display_field_name,
                    description=description,
                    field_info=field_info,
                )
                field_values[field_name] = _validate_and_store_field_value(
                    model_class, draft_model, field_name, value
                )

            return model_class(**field_values)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            questionary.print("\n⚠️ 入力値にエラーがあります:", style="bold orange")

            error_fields = set()

            if isinstance(e, ValidationError):
                field_errors: dict[str, list[str]] = {}
                has_model_level_error = False
                for error in e.errors():
                    loc = error.get("loc", ())
                    if loc and isinstance(loc[0], str) and loc[0] in fields:
                        field_name = loc[0]
                        error_fields.add(field_name)

                        index = loc[1] if len(loc) > 1 else None
                        display_field_name = (
                            f"{field_name}[{index}]"
                            if index is not None
                            else field_name
                        )
                    else:
                        has_model_level_error = True
                        display_field_name = parent or model_class.__name__

                    if display_field_name not in field_errors:
                        field_errors[display_field_name] = []
                    field_errors[display_field_name].append(error["msg"])

                for field_name, errors in field_errors.items():
                    questionary.print(f"  {field_name}:", style="bold")
                    for error_msg in errors:
                        print(f"    * {error_msg}")

                if has_model_level_error:
                    error_fields = set(fields.keys())

                for field_name, value in field_values.items():
                    if field_name not in error_fields:
                        valid_inputs[field_name] = value

                print("エラーとなった項目を再入力してください")
            else:
                print(f"エラーが発生しました: {str(e)}")
                print("すべての項目を再入力してください")
                valid_inputs = {}

            print()


class InputAborted(KeyboardInterrupt):
    def __str__(self):
        return "\n\n⚠️ 入力が中断されました。"


def scan_model_input(model_class: Type[T]) -> T:
    """
    モデルのフィールドをインタラクティブに入力し、モデルのインスタンスを返します。
    標準入力がリダイレクトされている場合はtar形式としてデータを読み込み、ファイル名／データを名前／データに変換します。

    フィールドがBaseModelを実装している場合には再帰的に処理します。
    BaseModelの配列のようなケースには未対応。

    Parameters
    ----------
    model_class
        Pydanticモデルクラス
    """
    try:
        return _scan_object(model_class, None)
    except KeyboardInterrupt:
        raise InputAborted() from None


def custom_input(input_method: Literal["text", "path"] = "text"):
    def wrapper(source_type):
        setattr(source_type, "__analysisrun_custom_input__", True)
        setattr(source_type, "__analysisrun_input_method__", input_method)
        return source_type

    return wrapper


def _is_custom_input(v) -> bool:
    return hasattr(v, "__analysisrun_custom_input__")


def _get_custom_input_method(v) -> Literal["text", "path"]:
    return getattr(v, "__analysisrun_input_method__", "text")


def _is_pydantic_model(v) -> TypeGuard[Type[BaseModel]]:
    return not _is_custom_input(v) and (
        is_dataclass(v) or hasattr(v, "__get_pydantic_core_schema__")
    )


@custom_input(input_method="path")
@deprecated("削除予定。VirtualFileを使用してください。")
class FilePath(str):
    """
    ファイルパスを表す文字列型。バリデーションの際にファイルの存在を確認します。
    文字列の前後にクォーテーションがある場合は削除します。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if not isinstance(v, str):
                return v
            if (v.startswith("'") and v.endswith("'")) or (
                v.startswith('"') and v.endswith('"')
            ):
                v = v[1:-1]
            p = Path(v)
            if not p.exists():
                raise ValueError(f"ファイルが存在しません: '{v}'")
            if not p.is_file():
                raise ValueError(f"ファイルのパスを指定してください: '{v}'")
            return v

        return core_schema.no_info_plain_validator_function(validate)


@custom_input(input_method="path")
class DirectoryPath(str):
    """
    ディレクトリパスを表す文字列型。バリデーションの際にディレクトリの存在を確認します。
    文字列の前後にクォーテーションがある場合は削除します。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if not isinstance(v, str):
                return v
            if (v.startswith("'") and v.endswith("'")) or (
                v.startswith('"') and v.endswith('"')
            ):
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


@custom_input(input_method="path")
class VirtualFile(Path):
    """
    仮想ファイルを扱う型。Pathオブジェクトまたはio.BytesIOオブジェクトを受け入れます。
    Pathオブジェクトのようにふるまうほか、io.BytesIOオブジェクトが与えられた場合にはファイルのように振る舞います。
    pandasのread_csvで読み込みができることを確認しています。

    Pathあるいは文字列が与えられた場合:
    - ファイルの存在を確認し、存在しなければエラーとなります。

    io.BytesIOが与えられた場合:
    - 作業ディレクトリに存在する仮想ファイルのように振る舞います。
    - ファイル名は仮想的に"virtual-file"とします。
    """

    def __init__(self, v: Path | BytesIO):
        if isinstance(v, Path):
            super().__init__(v)
        elif isinstance(v, BytesIO):
            super().__init__(Path(getcwd()) / "virtual-file")
            self._buffer = v

            # io.BytesIOが与えられた場合のみ、FileLikeオブジェクトとして振る舞うためのメソッドを追加
            def read(self: VirtualFile | None = None) -> bytes:
                return v.read()

            def iter(self: VirtualFile | None = None):
                raise NotImplementedError()

            self.read = read
            self.__iter__ = iter

    def unwrap(self) -> BytesIO | Path:
        """
        VirtualFileが内部的に保持しているio.BytesIOオブジェクトまたはPathオブジェクトを返します。
        """
        if hasattr(self, "_buffer"):
            return self._buffer
        return self

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate(v):
            if isinstance(v, (str, Path)):
                if isinstance(v, str):
                    if (v.startswith("'") and v.endswith("'")) or (
                        v.startswith('"') and v.endswith('"')
                    ):
                        v = v[1:-1]
                    v = Path(v)
                try:
                    if not v.exists():
                        raise ValueError(f"ファイルが存在しません: '{v}'")
                    if not v.is_file():
                        raise ValueError(f"ファイルのパスを指定してください: '{v}'")
                except Exception:
                    raise ValueError(
                        f"ファイルパスにアクセスすることができません: '{v}'"
                    )
                return cls(v)
            elif isinstance(v, BytesIO):
                return cls(v)  # type: ignore
            raise TypeError(f"Unsupported type: {type(v)}")

        return core_schema.no_info_plain_validator_function(validate)
