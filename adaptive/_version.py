from pathlib import Path

import versioningit

REPO_ROOT = Path(__file__).parent.parent
__version__ = versioningit.get_version(project_dir=REPO_ROOT)
