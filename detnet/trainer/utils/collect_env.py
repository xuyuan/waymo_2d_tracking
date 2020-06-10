import os
import re
from torch.utils.collect_env import get_env_info, get_platform, run, run_and_read_all, pretty_str, SystemEnv, PY3

DEFAULT_PACKAGES = "numpy torch torchvision pillow cudatoolkit opencv scipy pretrainedmodels tqdm pandas pyyaml scikit-image pycocotools apex tensorboard"


def get_grep_cmd(packages=DEFAULT_PACKAGES):
    if get_platform() == 'win32':
        grep_cmd = r'findstr /R "' + packages + '"'
    else:
        grep_cmd = r'grep "' + r'\|'.join(packages.split()) + '"'
    return grep_cmd


def get_conda_packages(packages=DEFAULT_PACKAGES, run_lambda=run):
    grep_cmd = get_grep_cmd(packages)
    conda = os.environ.get('CONDA_EXE', 'conda')
    out = run_and_read_all(run_lambda, conda + ' list | ' + grep_cmd)
    if out is None:
        return out
    # Comment starting at beginning of line
    comment_regex = re.compile(r'^#.*\n')
    return re.sub(comment_regex, '', out)


def get_pip_packages(packages=DEFAULT_PACKAGES, run_lambda=run):
    # People generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        grep_cmd = get_grep_cmd(packages)
        return run_and_read_all(run_lambda, pip + ' list --format=freeze | ' + grep_cmd)

    if not PY3:
        return 'pip', run_with_pip('pip')

    # Try to figure out if the user is running pip or pip3.
    out2 = run_with_pip('pip')
    out3 = run_with_pip('pip3')

    num_pips = len([x for x in [out2, out3] if x is not None])
    if num_pips == 0:
        return 'pip', out2

    if num_pips == 1:
        if out2 is not None:
            return 'pip', out2
        return 'pip3', out3

    # num_pips is 2. Return pip3 by default b/c that most likely
    # is the one associated with Python 3
    return 'pip3', out3


def get_conda_or_pip_packages(packages=DEFAULT_PACKAGES):
    conda_packages = get_conda_packages(packages)
    if conda_packages:
        return 'conda_packages', conda_packages

    pip_packages = get_pip_packages(packages)[1]
    return 'pip_packages', pip_packages


def get_pretty_env_info(packages=None):
    env_info = get_env_info()
    env_info = env_info._asdict()

    pkg_manager, found_packages = get_conda_or_pip_packages()

    not_found = []
    if packages:
        pkg_found = []
        for p in packages:
            if pkg_manager == 'conda_packages':
                p_ret = get_conda_packages(p)
            else:
                p_ret = get_pip_packages(p)[1]
            if not p_ret:
                not_found.append('✘ ' + p)
            elif p not in DEFAULT_PACKAGES:
                pkg_found.append('\n' + p_ret)
        found_packages += ' '.join(pkg_found)

    env_info[pkg_manager] = found_packages

    env_info = SystemEnv(**env_info)
    p_str = pretty_str(env_info)
    if not_found:
        p_str += '\nRequired but not found:\n'
        p_str += '\n'.join(not_found)
    return p_str


def check_perf():
    "Suggest how to improve the setup to speed things up"
    from PIL import features, Image

    print("Running performance checks:")
    # Pillow-SIMD check
    try:
        pil_version = Image.__version__  # PIL >= 7
    except:
        pil_version = Image.PILLOW_VERSION  # PIL <  7
    if re.search(r'\.post\d+', pil_version):
        print(f"✔ Running Pillow-SIMD {pil_version}")
    else:
        print(f"✘ Running Pillow {pil_version}")

    # libjpeg_turbo check
    if features.check_feature('libjpeg_turbo'):
        print("✔ libjpeg-turbo is on")
    else:
        print("✘ libjpeg-turbo is not on.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packages', nargs='*', help='packages required')
    args = parser.parse_args()
    print("Collecting environment information...")
    output = get_pretty_env_info(args.packages)
    print(output)
    check_perf()
