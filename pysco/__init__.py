import sys, os

# Add pysco subdirectory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
del sys, os

from pysco.main import run
