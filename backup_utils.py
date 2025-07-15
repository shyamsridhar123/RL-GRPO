import os
import sys
import shutil
import datetime

def create_backup(source_dir, backup_name=None):
    """
    Creates a backup of the source directory with a timestamp
    
    Args:
        source_dir: The directory to backup
        backup_name: Optional custom name for the backup folder
    
    Returns:
        Path to the created backup directory
    """
    # Create timestamp for unique backup name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if backup_name is None:
        backup_name = f"backup_{timestamp}"
    else:
        backup_name = f"{backup_name}_{timestamp}"
    
    # Create backup directory in the archive folder
    archive_dir = os.path.join(os.path.dirname(source_dir), "archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    backup_dir = os.path.join(archive_dir, backup_name)
    
    # Copy the directory
    shutil.copytree(source_dir, backup_dir)
    print(f"Backup created at {backup_dir}")
    
    return backup_dir

if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        backup_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            backup_path = create_backup(source_dir, backup_name)
            print(f"Successfully backed up {source_dir} to {backup_path}")
        else:
            print(f"Error: {source_dir} is not a valid directory")
    else:
        print("Usage: python backup_utils.py <source_directory> [backup_name]")
