from pathlib import Path
HERE=Path(__file__).resolve().parent
SITE=HERE.parent
DATA=HERE/'data'
RESULTS=HERE/'results'
REFERENCE=HERE/'reference'
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
