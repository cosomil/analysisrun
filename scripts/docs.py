"""
指定したディレクトリのPythonモジュールを静的解析し、ドキュメントを生成する。

ルール:
- ドキュメントはモジュール毎に章を分割すること
- 空のモジュールはスキップする
- メソッドやメンバ変数はそれに属するクラスのグループとしてまとめる
- docstringはNumPyスタイルで記述されている
- "_"から始まる非公開関数や非公開メソッド、非公開メンバ変数はスキップする（ただしコンストラクタ __init__ は除く）
- 公開関数、公開クラス、公開メンバ変数は全て出力対象とする
- docstringをパースして、Markdown形式に変換する
"""

import os
import sys
import ast
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, field


def is_private(name: str) -> bool:
    """
    Check if a name represents a private element (starts with '_').
    Constructor (__init__) is treated as public.

    Parameters
    ----------
    name : str
        The name to check

    Returns
    -------
    bool
        True if the name starts with '_' (except for __init__), False otherwise
    """
    return name.startswith("_") and name != "__init__"


@dataclass
class ParameterInfo:
    """パラメータ情報を保持するデータクラス"""

    name: str
    type: str
    desc: List[str] = field(default_factory=list)


@dataclass
class ReturnInfo:
    """戻り値情報を保持するデータクラス"""

    type: str
    desc: List[str] = field(default_factory=list)


@dataclass
class DocstringInfo:
    """NumPyスタイルのdocstringの解析結果を保持するデータクラス"""

    description: str = ""
    parameters: List[ParameterInfo] = field(default_factory=list)
    returns: List[ReturnInfo] = field(default_factory=list)
    other_sections: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AttributeInfo:
    """クラス属性情報を保持するデータクラス"""

    type: str
    docstring: Optional[str] = None


@dataclass
class MethodInfo:
    """メソッド情報を保持するデータクラス"""

    docstring: Optional[str]
    node: ast.FunctionDef


@dataclass
class ConstructorInfo:
    """コンストラクタ情報を保持するデータクラス"""

    docstring: Optional[str]
    node: ast.FunctionDef


@dataclass
class ClassInfo:
    """クラス情報を保持するデータクラス"""

    docstring: Optional[str]
    constructor: Optional[ConstructorInfo] = None
    methods: Dict[str, MethodInfo] = field(default_factory=dict)
    bases: List[str] = field(default_factory=list)
    attributes: Dict[str, AttributeInfo] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """関数情報を保持するデータクラス"""

    docstring: Optional[str]
    node: ast.FunctionDef


@dataclass
class ModuleInfo:
    """モジュール情報を保持するデータクラス"""

    docstring: Optional[str]
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)


def parse_docstring(node: ast.AST) -> Optional[str]:
    """
    Extract docstring from an AST node.
    """
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
        return None

    if not ast.get_docstring(node):
        return None

    return ast.get_docstring(node)


def parse_module_file(file_path: str) -> ModuleInfo:
    """
    Parse a Python module file and extract its structure and docstrings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        print(f"Could not parse {file_path}: {e}", file=sys.stderr)
        return ModuleInfo(docstring=None)

    module_info = ModuleInfo(docstring=parse_docstring(tree))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and not is_private(node.name):
            class_info = ClassInfo(
                docstring=parse_docstring(node),
                bases=[ast.unparse(base).strip() for base in node.bases],
            )

            for class_node in ast.iter_child_nodes(node):
                if isinstance(class_node, ast.FunctionDef):
                    if class_node.name == "__init__":
                        # コンストラクタは特別に処理
                        class_info.constructor = ConstructorInfo(
                            docstring=parse_docstring(class_node),
                            node=class_node,
                        )
                    elif not is_private(class_node.name):
                        # 通常のメソッド
                        class_info.methods[class_node.name] = MethodInfo(
                            docstring=parse_docstring(class_node),
                            node=class_node,
                        )
                # クラスのメンバ変数を検出
                elif isinstance(class_node, ast.AnnAssign) and hasattr(
                    class_node.target, "id"
                ):
                    attr_name = class_node.target.id  # type: ignore
                    if not is_private(attr_name):
                        # 型アノテーションがある場合に取得
                        attr_type = ""
                        if class_node.annotation:
                            attr_type = ast.unparse(class_node.annotation).strip()

                        # docstringを探す (次のノードがExprのStringの場合)
                        attr_docstring = None
                        class_nodes = list(ast.iter_child_nodes(node))
                        idx = class_nodes.index(class_node)
                        if idx + 1 < len(class_nodes):
                            next_node = class_nodes[idx + 1]
                            if (
                                isinstance(next_node, ast.Expr)
                                and isinstance(next_node.value, ast.Constant)
                                and isinstance(next_node.value.value, str)
                            ):
                                attr_docstring = next_node.value.value

                        class_info.attributes[attr_name] = AttributeInfo(
                            type=attr_type,
                            docstring=attr_docstring,
                        )

            module_info.classes[node.name] = class_info

        elif isinstance(node, ast.FunctionDef) and not is_private(node.name):
            module_info.functions[node.name] = FunctionInfo(
                docstring=parse_docstring(node),
                node=node,
            )

    return module_info


def get_function_signature(node: ast.FunctionDef) -> str:
    """
    Extract function signature from AST node.

    Parameters
    ----------
    node : ast.FunctionDef
        Function definition node

    Returns
    -------
    str
        Formatted function signature
    """
    # Get arguments
    args = []
    for arg in node.args.args:
        # Skip 'self' parameter for methods
        if arg.arg == "self":
            continue

        arg_str = arg.arg
        if hasattr(arg, "annotation") and arg.annotation is not None:
            annotation = ast.unparse(arg.annotation).strip()
            arg_str += f": {annotation}"
        args.append(arg_str)

    # Get return type
    returns = ""
    if hasattr(node, "returns") and node.returns is not None:
        returns = f" -> {ast.unparse(node.returns).strip()}"

    return f"{node.name}({', '.join(args)}){returns}"


def parse_numpy_docstring(docstring: str) -> DocstringInfo:
    """
    Parse a NumPy style docstring into sections.

    Parameters
    ----------
    docstring : str
        The docstring to parse

    Returns
    -------
    DocstringInfo
        Parsed docstring sections
    """
    if not docstring:
        return DocstringInfo(description="")

    lines = docstring.splitlines()

    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    result = DocstringInfo()
    description_lines = []

    current_section = "description"
    current_param = None
    current_return = None
    param_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()

        # Check for section headers
        if (
            i < len(lines) - 1
            and lines[i + 1].strip()
            and set(lines[i + 1].strip()) <= {"-"}
        ):
            # This is a section header
            current_section = line_stripped
            if current_section == "Parameters":
                current_param = None
            elif current_section == "Returns":
                current_return = None
            else:
                if current_section not in result.other_sections:
                    result.other_sections[current_section] = []
            i += 2  # Skip the section header line and the underlining
            continue

        # Process content based on current section
        if current_section == "description":
            description_lines.append(line)
        elif current_section == "Parameters":
            if (
                line_stripped
                and not line.startswith(" " * 4)
                and line_stripped != "----------"
            ):
                # This is a parameter definition line
                parts = line_stripped.split(":", 1)
                param_name = parts[0].strip()
                param_type = parts[1].strip() if len(parts) > 1 else ""
                current_param = ParameterInfo(name=param_name, type=param_type, desc=[])
                result.parameters.append(current_param)
                param_indent = len(line) - len(line.lstrip())
            elif current_param and line.startswith(" " * (param_indent + 4)):
                # This is parameter description
                current_param.desc.append(line.strip())
        elif current_section == "Returns":
            if (
                line_stripped
                and not line.startswith(" " * 4)
                and line_stripped != "----------"
            ):
                # This is a return definition line
                parts = line_stripped.split(":", 1)
                return_type = parts[0].strip()
                current_return = ReturnInfo(type=return_type, desc=[])
                result.returns.append(current_return)
                param_indent = len(line) - len(line.lstrip())
            elif current_return and line.startswith(" " * (param_indent + 4)):
                # This is return description
                current_return.desc.append(line.strip())
        elif current_section in result.other_sections:
            if line_stripped != "----------":  # Skip the divider lines
                result.other_sections[current_section].append(line)

        i += 1

    # Join description lines
    result.description = "\n".join(description_lines).strip()

    return result


def format_docstring_as_markdown(
    docstring: str, function_node: Optional[ast.FunctionDef] = None
) -> str:
    """
    Format a NumPy style docstring to Markdown.

    Parameters
    ----------
    docstring : str
        The docstring to format
    function_node : Optional[ast.FunctionDef]
        関数定義のASTノード。関数のパラメータ型情報を取得するのに使用

    Returns
    -------
    str
        Markdown formatted docstring
    """
    if not docstring:
        return ""

    parsed = parse_numpy_docstring(docstring)
    result = []

    # Add description
    if parsed.description:
        result.append(parsed.description)
        result.append("")  # Empty line

    # Add parameters
    if parsed.parameters:
        result.append("**引数**")

        # ASTからパラメータの型情報を取得
        param_types = {}
        if function_node:
            for arg in function_node.args.args:
                if arg.arg == "self":  # Skip self
                    continue
                if hasattr(arg, "annotation") and arg.annotation is not None:
                    param_types[arg.arg] = ast.unparse(arg.annotation).strip()

        for param in parsed.parameters:
            param_name = param.name
            param_desc = " ".join(param.desc)

            # docstringの型情報よりもASTの型アノテーションを優先する
            param_type = param_types.get(param_name, param.type)
            if param_type:
                result.append(f"- `{param_name}: {param_type}`: {param_desc}")
            else:
                result.append(f"- `{param_name}`: {param_desc}")

        result.append("")  # Empty line

    # Add returns
    if parsed.returns:
        result.append("**戻り値**")

        # 戻り値の型情報をASTから取得
        return_type_from_ast = ""
        if (
            function_node
            and hasattr(function_node, "returns")
            and function_node.returns is not None
        ):
            return_type_from_ast = ast.unparse(function_node.returns).strip()

        for ret in parsed.returns:
            ret_desc = " ".join(ret.desc)
            # ASTの型情報を優先
            ret_type = return_type_from_ast or ret.type
            result.append(f"- `{ret_type}`: {ret_desc}")

        result.append("")  # Empty line

    # Add other sections
    for section, lines in parsed.other_sections.items():
        result.append(f"**{section}**")
        result.append("\n".join(lines))
        result.append("")  # Empty line

    return "\n".join(result).strip()


def format_constructor_docstring_as_markdown(
    docstring: str, function_node: Optional[ast.FunctionDef] = None
) -> str:
    """
    Format a NumPy style docstring to Markdown for constructors.
    Skip the description section and only show parameters.

    Parameters
    ----------
    docstring : str
        The docstring to format
    function_node : Optional[ast.FunctionDef]
        関数定義のASTノード。関数のパラメータ型情報を取得するのに使用

    Returns
    -------
    str
        Markdown formatted docstring (parameters only)
    """
    if not docstring:
        return ""

    parsed = parse_numpy_docstring(docstring)
    result = []

    # Skip description for constructors

    # Add parameters only
    if parsed.parameters:
        # ASTからパラメータの型情報を取得
        param_types = {}
        if function_node:
            for arg in function_node.args.args:
                if arg.arg == "self":  # Skip self
                    continue
                if hasattr(arg, "annotation") and arg.annotation is not None:
                    param_types[arg.arg] = ast.unparse(arg.annotation).strip()

        for param in parsed.parameters:
            param_name = param.name
            param_desc = " ".join(param.desc)

            # docstringの型情報よりもASTの型アノテーションを優先する
            param_type = param_types.get(param_name, param.type)
            if param_type:
                result.append(f"- `{param_name}: {param_type}`: {param_desc}")
            else:
                result.append(f"- `{param_name}`: {param_desc}")

        result.append("")  # Empty line

    return "\n".join(result).strip()


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory and its subdirectories.
    """
    py_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)
                py_files.append(file_path)

    return py_files


def generate_markdown(module_path: str, output_file: str) -> None:
    """
    Generate markdown documentation for a module and its submodules.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(module_path):
        print(f"{module_path} is not a directory.", file=sys.stderr)
        return

    module_name = os.path.basename(module_path)
    python_files = find_python_files(module_path)

    with open(output_file, "w", encoding="utf-8") as md_file:
        for file_path in python_files:
            rel_path = os.path.relpath(file_path, module_path)
            package_parts = os.path.splitext(rel_path)[0].split(os.sep)

            # Determine the module name from the file path
            if len(package_parts) == 1:
                # Top-level module
                file_module_name = f"{module_name}.{package_parts[0]}"
            else:
                # Submodule
                file_module_name = f"{module_name}.{'.'.join(package_parts)}"

            # Parse the module
            module_info = parse_module_file(file_path)

            if not module_info.classes and not module_info.functions:
                continue

            # Write module documentation to the file (見出しレベルを上げる)
            md_file.write(f"# {file_module_name}\n\n")

            if module_info.docstring:
                md_file.write(
                    f"{format_docstring_as_markdown(module_info.docstring or '')}\n\n"
                )

            # Document classes
            if module_info.classes:
                for class_name, class_info in module_info.classes.items():
                    if is_private(class_name):
                        continue

                    # 継承情報がある場合に表示
                    if class_info.bases:
                        bases_str = ", ".join(class_info.bases)
                        md_file.write(f"## `class {class_name}({bases_str})`\n\n")
                    else:
                        md_file.write(f"## `class {class_name}`\n\n")

                    if class_info.docstring:
                        md_file.write(
                            f"{format_docstring_as_markdown(class_info.docstring or '')}\n\n"
                        )

                    # Document constructor (only if it has a docstring)
                    if class_info.constructor and class_info.constructor.docstring:
                        constructor_info = class_info.constructor
                        md_file.write("**コンストラクタの引数**\n")
                        md_file.write(
                            f"{format_constructor_docstring_as_markdown(constructor_info.docstring or '', constructor_info.node)}\n\n"
                        )

                    # Document class attributes/members
                    if class_info.attributes:
                        md_file.write("**メンバ変数**\n\n")
                        for attr_name, attr_info in class_info.attributes.items():
                            attr_type = attr_info.type
                            attr_desc = ""
                            if attr_info.docstring:
                                attr_desc = attr_info.docstring.strip()

                            if attr_type:
                                md_file.write(
                                    f"- `{attr_name}: {attr_type}`: {attr_desc}\n"
                                )
                            else:
                                md_file.write(f"- `{attr_name}`: {attr_desc}\n")

                        md_file.write("\n")

                    # Document methods
                    if class_info.methods:
                        for method_name, method_info in class_info.methods.items():
                            if is_private(method_name):
                                continue

                            # Get method signature
                            method_sig = get_function_signature(method_info.node)
                            md_file.write(f"### `{class_name}.{method_sig}`\n\n")

                            if method_info.docstring:
                                md_file.write(
                                    f"{format_docstring_as_markdown(method_info.docstring or '', method_info.node)}\n\n"
                                )

            # Document functions (見出しレベルはそのまま)
            if module_info.functions:
                for func_name, func_info in module_info.functions.items():
                    if is_private(func_name):
                        continue

                    # Get function signature
                    func_sig = get_function_signature(func_info.node)
                    md_file.write(f"### `{func_sig}`\n\n")

                    if func_info.docstring:
                        md_file.write(
                            f"{format_docstring_as_markdown(func_info.docstring or '', func_info.node)}\n\n"
                        )

            # Add a separator between modules
            md_file.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown documentation for Python modules."
    )
    parser.add_argument("module_path", help="Path to the Python module directory")
    parser.add_argument(
        "--output",
        "-o",
        default="docs/documentation.md",
        help="Output file path for documentation",
    )

    args = parser.parse_args()

    generate_markdown(args.module_path, args.output)
    print(f"Documentation generated at {args.output}")


if __name__ == "__main__":
    main()
