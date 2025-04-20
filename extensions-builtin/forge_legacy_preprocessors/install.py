import os
from pathlib import Path
from typing import Optional, Tuple

import pkg_resources

import launch

repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package


def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f"install -U {package}",
                            f"forge_legacy_preprocessor requirement: {package_name}=={package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                        launch.run_pip(
                            f"install -U {package}",
                            f"forge_legacy_preprocessor requirement: {package_name}=={package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f"install {package}",
                        f"forge_legacy_preprocessor requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(f"Warning: Failed to install {package}, some preprocessors may not work.")


def try_install_from_wheel(pkg_name: str, wheel_url: str, version: Optional[str] = None):
    current_version = get_installed_version(pkg_name)
    if current_version is not None:
        if version is None:
            return
        if comparable_version(current_version) >= comparable_version(version):
            return

    try:
        launch.run_pip(
            f"install -U {wheel_url}",
            f"forge_legacy_preprocessor requirement: {pkg_name}",
        )
    except Exception as e:
        print(e)
        print(f"Warning: Failed to install {pkg_name}. Some processors will not work.")


install_requirements(main_req_file)

try_install_from_wheel(
    "handrefinerportable",
    wheel_url=os.environ.get(
        "HANDREFINER_WHEEL",
        "https://github.com/huchenlei/HandRefinerPortable/releases/download/v1.0.1/handrefinerportable-2024.2.12.0-py2.py3-none-any.whl",
    ),
    version="2024.2.12.0",
)
try_install_from_wheel(
    "depth_anything",
    wheel_url=os.environ.get(
        "DEPTH_ANYTHING_WHEEL",
        "https://github.com/huchenlei/Depth-Anything/releases/download/v1.0.0/depth_anything-2024.1.22.0-py2.py3-none-any.whl",
    ),
)
