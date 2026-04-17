import os


PROTECTED_FILES = {".gitignore", "requirements.txt", "README.md"}


def remove_empty_files(root_dir):
    for foldername, _subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in PROTECTED_FILES:
                continue

            file_path = os.path.join(foldername, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                try:
                    os.chmod(file_path, 0o666)
                    print(f"Deleting empty file: {file_path}")
                    os.remove(file_path)
                except Exception as exc:
                    print(f"Failed to delete {file_path}: {exc}")


remove_empty_files(".")
