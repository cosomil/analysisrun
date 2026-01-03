import tarfile
from io import BytesIO
from typing import Any, IO, Optional


def read_tar_as_dict(b: IO[bytes]) -> dict[str, Any]:
    """
    ストリームからtar形式のデータを読み込み、ファイル名をキー、内容を値とする辞書に変換して返す。

    エントリのPAXヘッダー"is_file"がセットされている場合はBytesIOとして、そうでない場合は文字列として扱う。

    エントリ名に「.」が含まれる場合は、ネストした辞書を構築する。
    例: エントリ "foo.bar" は {"foo": {"bar": value}} のような構造となる。

    Raises
    ------
    ValueError
        構造に矛盾がある場合（例: "foo"と"foo.bar"が両方存在する場合）
    """

    # tar形式として解析（直接ストリーム処理）
    result = {}
    with tarfile.open(fileobj=b, mode="r|") as tar:
        for member in tar:
            if member.isfile():
                # ファイルの内容を読み込む
                file_obj = tar.extractfile(member)
                if file_obj:
                    content = file_obj.read()

                    # PAXヘッダーの"is_file"をチェック
                    is_file = member.pax_headers.get("is_file")
                    if is_file:
                        # PAXヘッダーに"is_file"がある場合はBytesIOとして保存
                        value = BytesIO(content)
                    else:
                        # それ以外の場合は文字列として保存（UTF-8でデコード）
                        value = content.decode("utf-8").strip()

                    # エントリ名を「.」で分割してネストした辞書を構築
                    keys = member.name.split(".")
                    current = result

                    for i, key in enumerate(keys[:-1]):
                        if key not in current:
                            current[key] = {}
                        elif not isinstance(current[key], dict):
                            existing_path = ".".join(keys[: i + 1])
                            raise ValueError(
                                f"tar mapping error: {member.name} requires {existing_path} to be a dictionary, but {existing_path} already has a value"
                            )
                        current = current[key]

                    # ターゲットとなる辞書に値を設定
                    last_key = keys[-1]
                    if last_key in current:
                        if isinstance(current[last_key], dict):
                            existing_path = ".".join(keys)
                            raise ValueError(
                                f"tar mapping error: {member.name} requires {existing_path} to be a value, but {existing_path} is a dictionary"
                            )

                    # 既に同じキーが存在する場合は上書き（警告なし）
                    current[last_key] = value

    return result


def _encode_to_tar(tar: tarfile.TarFile, prefix: Optional[str], data: dict[str, Any]):
    for name, value in data.items():
        if value is None:
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        # 値をバイト列に変換
        is_file = False
        if isinstance(value, BytesIO):
            # BytesIOの場合はそのまま読み込む
            value.seek(0)  # 読み取り位置を先頭に戻す
            content = value.read()
            is_file = True
        elif isinstance(value, bytes):
            # bytesの場合はそのまま使用
            content = value
            is_file = True
        elif isinstance(value, dict):
            # ネストした辞書の場合は再帰的に処理
            _encode_to_tar(tar, full_name, value)
            continue
        else:
            # その他の場合は文字列に変換してエンコード
            content = str(value).encode("utf-8")

        # TarInfoオブジェクトを作成
        tar_info = tarfile.TarInfo(name=full_name)
        tar_info.size = len(content)

        # BytesIOの場合はPAXヘッダーに"is_file"をセット
        if is_file:
            tar_info.pax_headers = {"is_file": "true"}

        # tarアーカイブに追加
        tar.addfile(tar_info, BytesIO(content))


def create_tar_from_dict(data: dict[str, Any]) -> BytesIO:
    """
    辞書からtar形式のデータを作成し、BytesIOとして返す。
    read_tar_as_dictの逆操作。

    Parameters
    ----------
    data
        キーをメンバー名、値を内容とする辞書。
        値がBytesIOの場合はバイナリデータとして、
        それ以外の場合は文字列に変換してUTF-8エンコードする。
        値がNoneの場合にはスキップされる。

    Returns
    -------
    BytesIO
        tar形式のデータを含むBytesIOオブジェクト
    """
    tar_buffer = BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        _encode_to_tar(tar, None, data)

    # 読み取り位置を先頭に戻す
    tar_buffer.seek(0)
    return tar_buffer
