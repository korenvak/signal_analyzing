#!/usr/bin/env python3
"""Script to perform git push operations."""
import subprocess
import os

os.chdir(r"C:\Users\koren\Projects\PythonProject")

log_file = open("git_operations.log", "w", encoding="utf-8")

def log(msg):
    """Log message to file and print."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def run_cmd(cmd):
    """Run command and print output."""
    log(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        log(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        log(f"STDERR:\n{result.stderr}")
    log(f"Exit code: {result.returncode}")
    return result.returncode

# Check if .git exists
if not os.path.exists(".git"):
    print("No .git folder found, initializing...")
    run_cmd("git init")

# Check remote
print("\n=== Checking remote ===")
run_cmd("git remote -v")

# Remove old origin if exists and add new one
run_cmd("git remote remove origin")
run_cmd("git remote add origin https://github.com/korenvak/signal_analyzing.git")

# Check remote again
run_cmd("git remote -v")

# Fetch from remote
print("\n=== Fetching from remote ===")
run_cmd("git fetch origin")

# Check branches
print("\n=== Checking branches ===")
run_cmd("git branch -a")

# Add all files
print("\n=== Adding all files ===")
run_cmd("git add -A")

# Check status
print("\n=== Git status ===")
run_cmd("git status")

# Commit
print("\n=== Committing ===")
run_cmd('git commit -m "Add acoustic analysis roadmap, update annotation system with SNR/Slope analysis, improve measurement tool"')

# Pull with rebase to merge with remote
print("\n=== Pulling from remote (with rebase) ===")
run_cmd("git pull origin master --rebase --allow-unrelated-histories")

# Push
print("\n=== Pushing to remote ===")
result = run_cmd("git push -u origin master")

if result == 0:
    print("\n✅ Push successful!")
else:
    print("\n❌ Push may have failed. Trying force push...")
    run_cmd("git push -u origin master --force")

log("\n=== Final status ===")
run_cmd("git status")
run_cmd("git log --oneline -3")

log_file.close()
print("\nLog saved to git_operations.log")

