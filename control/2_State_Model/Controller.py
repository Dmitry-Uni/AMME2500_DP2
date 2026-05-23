"""Controller entry for 2_State_Model.

This module supports being run directly. When executed as a script it will
ensure the `control` package directory is on `sys.path` so sibling modules
like `Vehicle_Params` can be imported.
"""
import os
import sys

try:
    # Prefer package-relative imports when available
    from . import Plant  # type: ignore
except Exception:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import Reference_States  # type: ignore


def main():
    print("Running Controller main()")
    # Example usage: call Reference_States.main() if available
    if hasattr(Reference_States, 'main'):
        Reference_States.main()


if __name__ == '__main__':
    main()
