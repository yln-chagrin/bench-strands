#!/usr/bin/env python3
"""
清理脚本 - 清理 bench-strands 项目中的临时文件和日志

使用方法:
    python cleanup.py [--dry-run] [--keep-logs N]

选项:
    --dry-run       只显示将要删除的文件，不实际删除
    --keep-logs N   保留最近的 N 个日志文件 (默认: 3)
"""

import os
import glob
import argparse
from pathlib import Path
import shutil
from datetime import datetime

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent

def cleanup_performance_reports(dry_run=False):
    """清理性能报告文件"""
    root = get_project_root()
    patterns = [
        "final_performance_report_*.json",
        "demo_performance_report_*.json"
    ]
    
    deleted_files = []
    for pattern in patterns:
        files = list(root.glob(pattern))
        for file in files:
            if dry_run:
                print(f"[DRY RUN] 将删除: {file}")
            else:
                file.unlink()
                print(f"已删除: {file}")
            deleted_files.append(str(file))
    
    return deleted_files

def cleanup_backup_files(dry_run=False):
    """清理备份文件"""
    root = get_project_root()
    patterns = [
        "*.backup",
        "*.backup.*",
        "test_error_config.json.backup.*"
    ]
    
    deleted_files = []
    for pattern in patterns:
        files = list(root.glob(pattern))
        for file in files:
            if dry_run:
                print(f"[DRY RUN] 将删除: {file}")
            else:
                file.unlink()
                print(f"已删除: {file}")
            deleted_files.append(str(file))
    
    return deleted_files

def cleanup_temp_files(dry_run=False):
    """清理临时文件"""
    root = get_project_root()
    patterns = [
        "TASK*_COMPLETION_SUMMARY.md",
        "*.tmp",
        "*.temp"
    ]
    
    deleted_files = []
    for pattern in patterns:
        files = list(root.glob(pattern))
        for file in files:
            if dry_run:
                print(f"[DRY RUN] 将删除: {file}")
            else:
                file.unlink()
                print(f"已删除: {file}")
            deleted_files.append(str(file))
    
    return deleted_files

def cleanup_python_cache(dry_run=False):
    """清理Python缓存文件"""
    root = get_project_root()
    # 只清理项目根目录的缓存，排除虚拟环境
    cache_dirs = []
    for cache_dir in root.glob("**/__pycache__"):
        # 跳过虚拟环境目录
        if ".venv" not in str(cache_dir):
            cache_dirs.append(cache_dir)
    
    pytest_cache = root / ".pytest_cache"
    
    deleted_dirs = []
    
    # 清理 __pycache__ 目录
    for cache_dir in cache_dirs:
        if dry_run:
            print(f"[DRY RUN] 将删除目录: {cache_dir}")
        else:
            shutil.rmtree(cache_dir)
            print(f"已删除目录: {cache_dir}")
        deleted_dirs.append(str(cache_dir))
    
    # 清理 .pytest_cache 目录
    if pytest_cache.exists():
        if dry_run:
            print(f"[DRY RUN] 将删除目录: {pytest_cache}")
        else:
            shutil.rmtree(pytest_cache)
            print(f"已删除目录: {pytest_cache}")
        deleted_dirs.append(str(pytest_cache))
    
    return deleted_dirs

def cleanup_logs(keep_count=3, dry_run=False):
    """清理日志文件，保留最近的几个"""
    root = get_project_root()
    logs_dir = root / "logs"
    
    if not logs_dir.exists():
        return []
    
    # 按文件类型分组
    log_types = {
        "*.log": [],
        "*.jsonl": [],
        "*.json": []
    }
    
    for pattern in log_types.keys():
        files = list(logs_dir.glob(pattern))
        # 按修改时间排序，最新的在前
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        log_types[pattern] = files
    
    deleted_files = []
    
    # 对每种类型的文件，保留最近的 keep_count 个
    for pattern, files in log_types.items():
        files_to_delete = files[keep_count:]
        for file in files_to_delete:
            if dry_run:
                print(f"[DRY RUN] 将删除日志: {file}")
            else:
                file.unlink()
                print(f"已删除日志: {file}")
            deleted_files.append(str(file))
    
    return deleted_files

def main():
    parser = argparse.ArgumentParser(description="清理 bench-strands 项目中的临时文件")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要删除的文件，不实际删除")
    parser.add_argument("--keep-logs", type=int, default=3, help="保留最近的 N 个日志文件")
    
    args = parser.parse_args()
    
    print("🧹 开始清理 bench-strands 项目...")
    print(f"模式: {'预览模式 (不会实际删除)' if args.dry_run else '实际删除模式'}")
    print(f"保留日志数量: {args.keep_logs}")
    print("-" * 50)
    
    total_deleted = []
    
    # 清理性能报告
    print("\n📊 清理性能报告文件...")
    deleted = cleanup_performance_reports(args.dry_run)
    total_deleted.extend(deleted)
    
    # 清理备份文件
    print("\n💾 清理备份文件...")
    deleted = cleanup_backup_files(args.dry_run)
    total_deleted.extend(deleted)
    
    # 清理临时文件
    print("\n🗂️  清理临时文件...")
    deleted = cleanup_temp_files(args.dry_run)
    total_deleted.extend(deleted)
    
    # 清理Python缓存
    print("\n🐍 清理Python缓存...")
    deleted = cleanup_python_cache(args.dry_run)
    total_deleted.extend(deleted)
    
    # 清理日志文件
    print(f"\n📝 清理日志文件 (保留最近 {args.keep_logs} 个)...")
    deleted = cleanup_logs(args.keep_logs, args.dry_run)
    total_deleted.extend(deleted)
    
    print("\n" + "=" * 50)
    print(f"✅ 清理完成！")
    print(f"{'预计' if args.dry_run else '实际'}处理文件/目录数量: {len(total_deleted)}")
    
    if args.dry_run:
        print("\n💡 要实际执行清理，请运行: python cleanup.py")
    else:
        print(f"\n📅 清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()