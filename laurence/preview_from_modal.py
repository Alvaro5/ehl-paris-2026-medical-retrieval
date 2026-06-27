"""Preview data from Modal mount ehl-2026-vol-2"""

import os
from pathlib import Path
import modal

app = modal.App("ehl-preview")

# Reference the shared volume mount
vol = modal.Volume.from_name("ehl-2026-vol-2")


@app.function(volumes={"/data": vol})
def preview_mount():
    """Preview files and structure from the ehl-2026-vol-2 mount"""
    mount_path = Path("/data")

    if not mount_path.exists():
        print(f"Mount path {mount_path} does not exist")
        return

    print(f"\n📁 Previewing mount: ehl-2026-vol-2")
    print(f"📍 Mount path: {mount_path}")
    print("=" * 60)

    # Walk through directory structure
    total_files = 0
    total_dirs = 0

    for root, dirs, files in os.walk(mount_path):
        level = root.replace(str(mount_path), "").count(os.sep)
        indent = " " * 2 * level
        rel_path = os.path.relpath(root, mount_path)

        if rel_path == ".":
            print("\n📂 Root contents:")
        else:
            print(f"\n{indent}📂 {rel_path}/")

        total_dirs += len(dirs)

        # Show files in this directory
        for file in files[:10]:  # Limit to first 10 files per dir
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            size_str = format_size(file_size)
            print(f"{indent}  📄 {file} ({size_str})")
            total_files += 1

        if len(files) > 10:
            print(f"{indent}  ... and {len(files) - 10} more files")
            total_files += len(files) - 10
        else:
            total_files += len(files)

    print("\n" + "=" * 60)
    print(f"✅ Summary: {total_dirs} directories, {total_files} files found")


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"


@app.local_entrypoint()
def main():
    """Main entry point"""
    preview_mount.remote()


if __name__ == "__main__":
    main()
