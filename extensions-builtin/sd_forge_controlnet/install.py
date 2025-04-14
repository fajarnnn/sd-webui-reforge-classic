from pathlib import Path
import launch


repo_root = Path(__file__).parent
main_req_file = repo_root.joinpath("requirements.txt")


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package


def install_requirements(req_file: str):
    with open(req_file) as file:
        packages = [pkg.strip() for pkg in file.readlines()]

    for package in packages:
        try:
            if not launch.is_installed(extract_base_package(package)):
                launch.run_pip(
                    f"install {package}",
                    f"sd-forge-controlnet requirement: {package}",
                )
        except Exception as e:
            print(f"Warning: Failed to install {package}\n{e}")


install_requirements(main_req_file)
