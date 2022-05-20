from ast import Import


import sys
import typer
from wasabi import msg

try:
    import prodigy
except ImportError:
    msg.fail("No installation of prodigy found")
    sys.exit()
