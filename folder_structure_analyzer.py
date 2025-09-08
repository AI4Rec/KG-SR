import os
from pathlib import Path
import sys
from datetime import datetime
import io
import tokenize
from fnmatch import fnmatch

# ============ 基础输出 ============

class DualOutput:
    """双重输出：同时输出到控制台和文件"""
    def __init__(self, output_file):
        self.output_file = output_file
        self.file_handle = None
    
    def __enter__(self):
        self.file_handle = open(self.output_file, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.file_handle:
            print(*args, **kwargs, file=self.file_handle)

# ============ 忽略目录规则 ============

def should_ignore_dir(name: str, ignore_dirs_lower: set, ignore_patterns: set) -> bool:
    """
    判断目录名是否应被忽略：
    - 大小写不敏感的精确匹配（ignore_dirs_lower）
    - 通配符模式（ignore_patterns，大小写不敏感）
    """
    n = name.lower()
    if n in ignore_dirs_lower:
        return True
    # 通配符匹配不自带大小写不敏感，这里将模式和目标都降为小写再比
    for pat in ignore_patterns:
        if fnmatch(n, pat.lower()):
            return True
    return False

# ============ 树形结构 ============

def print_tree_structure(
    root_path, 
    prefix="", 
    ignore_dirs=None, 
    ignore_patterns=None, 
    is_last=True, 
    output=None
):
    # 默认忽略（统一小写）
    if ignore_dirs is None:
        ignore_dirs = {"saved", "scripts", "bert", "__pycache__"}
    if ignore_patterns is None:
        ignore_patterns = {"saved*", "*scripts*", "bert*"}
    ignore_dirs_lower = {d.lower() for d in ignore_dirs}

    root_path = Path(root_path)
    # 若根节点本身命中忽略，直接返回（一般不会把忽略目录当根）
    if root_path.is_dir() and should_ignore_dir(root_path.name, ignore_dirs_lower, ignore_patterns):
        return

    connector = "└── " if is_last else "├── "
    line = f"{prefix}{connector}{root_path.name}"
    (output.print if output else print)(line)
    
    if root_path.is_file():
        return
    
    try:
        children = []
        for c in root_path.iterdir():
            if c.is_dir() and should_ignore_dir(c.name, ignore_dirs_lower, ignore_patterns):
                continue
            children.append(c)
        # 目录优先
        children.sort(key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return
    
    extension = "    " if is_last else "│   "
    new_prefix = prefix + extension
    
    for i, child in enumerate(children):
        print_tree_structure(
            child, 
            new_prefix, 
            ignore_dirs=ignore_dirs, 
            ignore_patterns=ignore_patterns, 
            is_last=(i == len(children) - 1), 
            output=output
        )

# ============ 工具函数 ============

def is_binary_file(file_path):
    binary_extensions = {'.pth', '.tar', '.pkl', '.npy', '.npz', '.pyc', '.pt',
                         '.so', '.dll', '.exe', '.bin', '.dat'}
    file_path = Path(file_path)
    if file_path.suffix.lower() in binary_extensions:
        return True
    if len(file_path.suffixes) > 1:
        combined = ''.join(file_path.suffixes).lower()
        if any(ext in combined for ext in ['.pth.tar', '.pkl.gz']):
            return True
    return False


def read_file_content(file_path, encoding_list=None):
    if encoding_list is None:
        encoding_list = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
    
    for enc in encoding_list:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return f"读取文件时发生错误: {e}"
    return "无法使用常见编码读取文件"


def _normalize_newlines_and_blank_lines(text: str) -> str:
    """统一换行并压缩连续空行为单个空行，去除行尾空白。"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [ln.rstrip() for ln in text.split('\n')]
    cleaned = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        cleaned.append(ln)
    # 末尾留一个换行，阅读更舒服
    return ("\n".join(cleaned)).rstrip() + "\n"


def strip_py_comments_and_docstrings(source_code: str) -> str:
    """
    去掉 .py 文件中的注释与 docstring。
    首选 AST 重建（能彻底移除注释/文档字符串且避免奇怪换行），
    若失败则回退到 tokenize 方案，并做后处理以减少空行与“\ + 换行”现象。
    """
    # ---- 方案一：AST ----
    try:
        import ast

        class _RmDoc(ast.NodeTransformer):
            def _strip_doc(self, node):
                if node.body and isinstance(node.body[0], ast.Expr):
                    val = node.body[0].value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        node.body = node.body[1:]
                return node

            def visit_Module(self, node):
                self.generic_visit(node)
                return self._strip_doc(node)

            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                return self._strip_doc(node)

            def visit_AsyncFunctionDef(self, node):
                self.generic_visit(node)
                return self._strip_doc(node)

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                return self._strip_doc(node)

        tree = ast.parse(source_code)
        tree = _RmDoc().visit(tree)
        ast.fix_missing_locations(tree)
        try:
            # Python 3.9+
            code = ast.unparse(tree)
        except Exception:
            # 某些环境可能没有 ast.unparse
            raise
        # AST 输出本身不含注释；再做统一换行与空行压缩
        return _normalize_newlines_and_blank_lines(code)
    except Exception:
        # ---- 方案二：tokenize 兜底 ----
        try:
            io_obj = io.StringIO(source_code)
            out = io.StringIO()
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0

            for tok in tokenize.generate_tokens(io_obj.readline):
                tok_type, tok_str, start, end, line = tok

                # 跳过 # 注释
                if tok_type == tokenize.COMMENT:
                    continue

                # 跳过独立成句的字符串（docstring）
                if tok_type == tokenize.STRING:
                    if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE):
                        if start[1] == 0:
                            prev_toktype = tok_type
                            continue

                if start[0] > last_lineno:
                    last_col = 0
                if start[1] > last_col:
                    out.write(" " * (start[1] - last_col))
                out.write(tok_str)
                prev_toktype = tok_type
                last_lineno, last_col = end

            cleaned = out.getvalue()
            # 统一行尾与空行；尽量避免奇怪的“\ + 换行”表现
            cleaned = _normalize_newlines_and_blank_lines(cleaned)
            return cleaned
        except Exception:
            # 实在失败就原样返回
            return source_code

# ============ 内容打印 ============

def print_file_content(file_path, file_type="full", output=None):
    file_path = Path(file_path)
    sep = f"\n{'='*60}"
    loc = f"文件位置: {file_path}"
    (output.print if output else print)(sep)
    (output.print if output else print)(loc)
    (output.print if output else print)("="*60)
    
    content = read_file_content(file_path)
    if content is None or (isinstance(content, str) and "错误" in content):
        (output.print if output else print)(content if content else "无法读取文件内容")
        return
    
    # 对 .py 文件做清洗
    if file_path.suffix.lower() == ".py":
        content = strip_py_comments_and_docstrings(content)
    
    if file_type == "full":
        (output.print if output else print)(content)
    else:
        lines = content.splitlines()
        for i, line in enumerate(lines[:5]):
            (output.print if output else print)(f"{i+1:3d}: {line}")
        if len(lines) > 5:
            (output.print if output else print)(f"... (文件共 {len(lines)} 行，仅显示前5行)")

# ============ 主流程 ============

def analyze_folder(target_folder, output_file=None):
    target_path = Path(target_folder)
    if not target_path.exists():
        print(f"错误: 路径 '{target_folder}' 不存在")
        return
    if not target_path.is_dir():
        print(f"错误: '{target_folder}' 不是一个文件夹")
        return
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = target_path.name if target_path.name else "root"
        output_file = f"folder_analysis_{folder_name}.txt"

    # 解析“本脚本路径”和“输出文件路径”，用于排除
    try:
        self_script_path = Path(__file__).resolve()
    except NameError:
        # 某些打包/嵌入环境可能没有 __file__
        self_script_path = Path(sys.argv[0]).resolve()
    output_file_path = Path(output_file).resolve()

    # 统一的忽略配置（可按需修改/扩展）
    ignore_dirs = {"scripts", "bert",  ".git","__pycache__"}     # 精确名（大小写不敏感）
    ignore_patterns = {"saved*", ".git*", "*scripts*", "bert*"}            # 通配

    ignore_dirs_lower = {d.lower() for d in ignore_dirs}

    with DualOutput(output_file_path) as output:
        output.print(f"分析文件夹: {target_path.absolute()}")
        output.print(f"输出文件: {output_file_path}")
        output.print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.print("="*60)
        
        output.print("\n文件夹结构:")
        print_tree_structure(
            target_path, 
            output=output, 
            ignore_dirs=ignore_dirs, 
            ignore_patterns=ignore_patterns
        )
        
        output.print("\n\n" + "="*60)
        output.print("文件内容分析:")
        output.print("="*60)
        
        def process_dir(cur: Path):
            try:
                for item in cur.iterdir():
                    # ---- 目录处理：命中忽略则跳过整个子树 ----
                    if item.is_dir():
                        if should_ignore_dir(item.name, ignore_dirs_lower, ignore_patterns):
                            continue
                        process_dir(item)
                        continue

                    # ---- 文件过滤：排除本脚本与输出文件自身 ----
                    try:
                        item_real = item.resolve()
                    except Exception:
                        item_real = item

                    if item_real == self_script_path or item_real == output_file_path:
                        continue

                    # 二进制直接跳过
                    if is_binary_file(item_real):
                        continue

                    suffix = item_real.suffix.lower()
                    if suffix in ('.yaml', '.yml', '.py'):
                        print_file_content(item_real, "full", output)
                    elif suffix in ('.csv', '.inter', '.kg', '.link', '.txt', '.md', '.json', '.log'):
                        print_file_content(item_real, "partial", output)
            except PermissionError:
                output.print(f"权限错误: 无法访问 {cur}")
        
        process_dir(target_path)
        output.print("\n" + "="*60)
        output.print("分析完成!")
        output.print(f"结果已保存到: {output_file_path}")
    
    print(f"\n✓ 分析完成！结果已保存到: {output_file_path}")

# ============ CLI ============

def main():
    if len(sys.argv) < 2:
        print("使用方法: python folder_structure_analyzer.py <目标文件夹路径> [输出文件路径]")
        return
    target = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_folder(target, output_file)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("未指定目标文件夹，将分析当前目录...")
        analyze_folder(".")
    else:
        main()