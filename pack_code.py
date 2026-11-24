import os
import shutil


def build_dir(path: str) -> str:
    """安全地创建目录，如果已存在则不报错"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def smart_copy(
    src: str,
    dst: str,
    ignore_names: list = None,
    ignore_exts: list = None,
) -> None:
    """
    复制整个目录，跳过 ignore_names（文件名/文件夹名）和 ignore_exts（后缀）中指定的内容
    """
    ignore_names = set(ignore_names) if ignore_names else set()
    ignore_exts = set(ignore_exts) if ignore_exts else set()

    def _ignore_func(directory, contents):
        ignored = set()
        for item in contents:
            # 匹配完整名称（文件或文件夹）
            if item in ignore_names:
                ignored.add(item)
                continue
            # 匹配后缀（仅对文件有效）
            if os.path.isfile(os.path.join(directory, item)):
                _, ext = os.path.splitext(item)
                if ext in ignore_exts:
                    ignored.add(item)
        return ignored

    if os.path.exists(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst, ignore=_ignore_func)
    print(f"[完成] {src} -> {dst}")


def main(out_dir: str):
    build_dir(out_dir)

    # ==========================================================
    # 📝 忽略规则配置
    # ==========================================================

    # 全局忽略的文件名 / 文件夹名
    global_ignore_names = [
        '.git',
        '.github',
        'runs',
        'Thumbs.db',
        '__pycache__',
        # 👇 在这里添加你想跳过的文件或文件夹名称
        # 例如: '.git', 'output', 'logs', 'pack_code.py'
    ]

    # 全局忽略的文件后缀
    global_ignore_exts = [
        '.pth',
        '.pt',
        '.onnx',
        '.log',
        # 👇 在这里添加你想跳过的后缀
        # 例如: '.bak', '.tmp'
    ]

    # ==========================================================
    # 🚀 复制整个项目目录（脚本所在目录）
    # ==========================================================
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 避免输出目录嵌套在源目录里导致无限递归
    abs_out = os.path.abspath(out_dir)
    if abs_out.startswith(project_root):
        # 把输出目录名加入忽略列表，防止 copytree 复制自己
        rel = os.path.relpath(abs_out, project_root)
        top_folder = rel.split(os.sep)[0]
        if top_folder not in global_ignore_names:
            global_ignore_names.append(top_folder)

    smart_copy(
        src=project_root,
        dst=out_dir,
        ignore_names=global_ignore_names,
        ignore_exts=global_ignore_exts,
    )


if __name__ == '__main__':
    out_dir = r'F:\project_distill\code\RT-DETRv4_pack_1124'
    main(out_dir)