try:
    import pytest
    print("pytest successfully imported")
except ImportError as e:
    print(f"Error importing pytest: {e}")

try:
    import streamlit
    print("streamlit successfully imported")
except ImportError as e:
    print(f"Error importing streamlit: {e}")

import sys
print("\nPython path:")
for path in sys.path:
    print(path)
