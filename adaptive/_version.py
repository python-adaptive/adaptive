import sys
import subprocess
import os

# No public API
__all__ = []

package_root = os.path.dirname(os.path.realpath(__file__))
distr_root = os.path.dirname(package_root)

STATIC_VERSION_FILE = '_adaptive_version.py'

version = None

def get_version(version_file=STATIC_VERSION_FILE):
    version_info = {}
    with open(os.path.join(package_root, version_file), 'rb') as f:
        exec(f.read(), {}, version_info)
    version = version_info['version']
    version_is_from_git = (version == "__use_git__")
    if version_is_from_git:
        version = get_version_from_git()
        if not version:
            version = get_version_from_git_archive(version_info)
        if not version:
            version = "unknown"
    return version


def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    if not os.path.samefile(p.communicate()[0].decode().rstrip('\n'),
                            distr_root):
        # The top-level directory of the current Git repository is not the same
        # as the root directory of the Kwant distribution: do not extract the
        # version from Git.
        return

    # git describe --first-parent does not take into account tags from branches
    # that were merged-in.
    for opts in [['--first-parent'], []]:
        try:
            p = subprocess.Popen(['git', 'describe', '--long'] + opts,
                                 cwd=distr_root,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    description = p.communicate()[0].decode().strip('v').rstrip('\n')

    release, dev, git = description.rsplit('-', 2)
    version = [release]
    labels = []
    if dev != "0":
        version.append(".dev{}".format(dev))
        labels.append(git)

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=distr_root)
    except OSError:
        labels.append('confused') # This should never happen.
    else:
        if p.wait() == 1:
            labels.append('dirty')

    if labels:
        version.append('+')
        version.append(".".join(labels))

    return "".join(version)


# TODO: change this logic when there is a git pretty-format
#       that gives the same output as 'git describe'.
#       Currently we return just the tag if a tagged version
#       was archived, or 'unknown-g<git hash>' otherwise.
def get_version_from_git_archive(version_info):
    try:
        refnames = version_info['refnames']
        git_hash = version_info['git_hash']
    except KeyError:
        # These fields are not present if we are running from an sdist.
        # Execution should never reach here, though
        return None

    if git_hash.startswith('$Format') or refnames.startswith('$Format'):
        # variables not expanded during 'git archive'
        return None

    TAG = 'tag: v'
    refs = set(r.strip() for r in refnames.split(","))
    tags = set(r[len(TAG):] for r in refs if r.startswith(TAG))
    if tags:
        release, *_ = sorted(tags)  # prefer e.g. "2.0" over "2.0rc1"
        return release
    else:
        return f'unknown+g{git_hash}'

version = get_version()
