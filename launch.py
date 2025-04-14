from modules import launch_utils

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

if args.uv or args.uv_symlink:
    from modules_forge.uv_hook import patch
    patch(args.uv_symlink)

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
start = launch_utils.start


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()
        print(f"Sysinfo saved as {filename}. Exiting...")
        exit(0)

    if args.uv or args.uv_symlink:
        print("using uv")

    launch_utils.startup_timer.record("initial startup")

    if not args.skip_prepare_environment:
        with launch_utils.startup_timer.subcategory("prepare environment"):
            prepare_environment()

    if args.forge_ref_a1111_home:
        launch_utils.configure_forge_reference_checkout(args.forge_ref_a1111_home)

    start()


if __name__ == "__main__":
    main()
