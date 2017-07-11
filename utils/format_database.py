import subprocess
import sys
import os.path
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imageretrieval.py path_to_database")
        sys.exit(1);