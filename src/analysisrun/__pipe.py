import tarfile
from io import BytesIO
from typing import Any, BinaryIO


def read_tar_as_dict(b: BinaryIO) -> dict[str, Any]:
    """
    ストリームからtar形式のデータを読み込み、ファイル名をキー、内容を値とする辞書に変換して返します。

    メンバー名に「.」が含まれる場合（拡張子がある場合）はBytesIOとして、含まれない場合は文字列として扱います。
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

                    # メンバー名に「.」が含まれるかチェック（拡張子の有無）
                    if "." in member.name:
                        # 拡張子がある場合はBytesIOとして保存
                        # 名前からは拡張子を除去する
                        name = member.name.rsplit(".", 1)[0]
                        result[name] = BytesIO(content)
                    else:
                        # 拡張子がない場合は文字列として保存（UTF-8でデコード）
                        result[member.name] = content.decode("utf-8").strip()

    return result


def create_tar_from_dict(data: dict[str, Any]) -> BytesIO:
    """
    辞書からtar形式のデータを作成し、BytesIOとして返します。
    read_and_transform_stdinの逆操作です。

    Parameters
    ----------
    data
        キーをメンバー名、値を内容とする辞書。
        値がBytesIOの場合はバイナリデータとして、
        それ以外の場合は文字列に変換してUTF-8エンコードします。

    Returns
    -------
    BytesIO
        tar形式のデータを含むBytesIOオブジェクト
    """
    tar_buffer = BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for name, value in data.items():
            # 値をバイト列に変換
            if isinstance(value, BytesIO):
                # BytesIOの場合はそのまま読み込む
                value.seek(0)  # 読み取り位置を先頭に戻す
                content = value.read()
            elif isinstance(value, bytes):
                # bytesの場合はそのまま使用
                content = value
            else:
                # その他の場合は文字列に変換してエンコード
                content = str(value).encode("utf-8")

            # TarInfoオブジェクトを作成
            tar_info = tarfile.TarInfo(name=name)
            tar_info.size = len(content)

            # tarアーカイブに追加
            tar.addfile(tar_info, BytesIO(content))

    # 読み取り位置を先頭に戻す
    tar_buffer.seek(0)
    return tar_buffer
