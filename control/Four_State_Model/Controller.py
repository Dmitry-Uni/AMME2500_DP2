import os
import sys
import numpy as np

try:
    # Prefer package-relative imports when available
    from .. import Reference_States  # type: ignore
except Exception:
    # Fallback for direct execution: add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from control import Reference_States  # type: ignore

try:
    # Prefer package-relative imports when available
    from .. import Vehicle_Params  # type: ignore
except Exception:
    # Fallback for direct execution: add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from control import Vehicle_Params  # type: ignore

# Main function to demonstrate usage
def main():
    print("Running ")


if __name__ == '__main__':
    main()