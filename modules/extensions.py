from __future__ import annotations

import configparser
import os
import threading
import re

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401
from modules_forge.config import always_disabled_extensions


os.makedirs(extensions_dir, exist_ok=True)
extensions: list["Extension"] = []


def active():
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class ExtensionMetadata:
    filename = "metadata.ini"
    config: configparser.ConfigParser
    canonical_name: str
    requires: list[str]

    def __init__(self, path, folder_name):
        filepath = os.path.join(path, self.filename)

        try:
            self.config = configparser.ConfigParser()
            self.config.read(filepath)
        except Exception:
            # `self.config.read()` catches `FileNotFoundError` automatically
            errors.report(f"Error reading {self.filename} for extension {folder_name}", exc_info=True)

        canonical_name = self.config.get("Extension", "Name", fallback=folder_name)
        self.canonical_name = canonical_name.lower().strip()
        self.requires = self.get_script_requirements("Requires", "Extension")

    def get_script_requirements(self, field, section, extra_section=None):
        """
        reads a list of requirements from the config; `field` is the name of the field in the `.ini` file,
        like **Requires** or **Before**, and `section` is the name of the `[section]` in the `.ini` file;
        additionally, reads more requirements from `[extra_section]` if specified
        """

        x = self.config.get(section, field, fallback="")
        if extra_section:
            x = ",".join([x, self.config.get(extra_section, field, fallback="")])

        return self.parse_list(x.lower())

    def parse_list(self, text):
        """
        converts a line from config `"ext1 ext2, ext3  "` into a python list `["ext1", "ext2", "ext3"]`
        """

        if not text:
            return []

        return [x.strip() for x in re.split(r"[,\s]+", text) if x.strip()]


class Extension:
    lock = threading.Lock()
    metadata: ExtensionMetadata
    cached_fields = ("remote", "commit_date", "branch", "commit_hash", "version")

    def __init__(self, name, path, enabled=True, is_builtin=False, metadata=None):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.is_builtin = is_builtin
        self.metadata = metadata

        self.status = ""
        self.can_update = False
        self.commit_hash = ""
        self.commit_date = None
        self.version = ""
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        self.canonical_name = metadata.canonical_name

    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                self.do_read_info_from_repo()
                return self.to_dict()

        try:
            d = cache.cached_data_for_file("extensions-git", self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass

        self.status = "unknown" if self.status == "" else self.status

    def do_read_info_from_repo(self):
        repo = None
        if os.path.exists(os.path.join(self.path, ".git")):
            try:
                repo = Repo(self.path)
            except Exception:
                errors.report(f"Error reading github repository info from {self.path}", exc_info=True)
                repo = None

        self.have_info_from_repo = True

        if repo is None or repo.bare:
            self.remote = None
            return

        try:
            self.remote = next(repo.remote().urls, None)
            commit = repo.head.commit
            self.commit_date = commit.committed_date
            if repo.active_branch:
                self.branch = repo.active_branch.name
            self.commit_hash = commit.hexsha
            self.version = self.commit_hash[:8]

        except Exception:
            errors.report(f"Error reading extension data ({self.name})", exc_info=True)
            self.remote = None

    def list_files(self, subdir, extension):
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        return [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    def check_updates(self):
        repo = Repo(self.path)
        for fetch in repo.remote().fetch(dry_run=True):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse("origin")
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit="origin"):
        # Fix: `error: Your local changes to the following files would be overwritten by merge`
        repo = Repo(self.path)
        repo.git.fetch(all=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False


def list_extensions():
    extensions.clear()
    loaded_extensions = {}

    if shared.cmd_opts.disable_all_extensions:
        print('*** "--disable-all-extensions" arg was used, will not load any extensions ***')
        return
    if shared.opts.disable_all_extensions == "all":
        print('*** "Disable all extensions" option was set, will not load any extensions ***')
        return

    if shared.cmd_opts.disable_extra_extensions:
        print('*** "--disable-extra-extensions" arg was used, will only load built-in extensions ***')
    elif shared.opts.disable_all_extensions == "extra":
        print('*** "Disable all extensions" option was set, will only load built-in extensions ***')

    disabled_extensions = shared.opts.disabled_extensions + always_disabled_extensions

    for dirname in (extensions_builtin_dir, extensions_dir):
        if not os.path.isdir(dirname):
            continue

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            canonical_name = extension_dirname
            metadata = ExtensionMetadata(path, canonical_name)
            is_builtin = dirname == extensions_builtin_dir

            already_loaded_extension = loaded_extensions.get(metadata.canonical_name)
            if already_loaded_extension is not None:
                errors.report(f'Duplicated name "{canonical_name}" found in extensions "{extension_dirname}" and "{already_loaded_extension.name}". Former will be discarded!', exc_info=False)
                continue

            extension = Extension(name=extension_dirname, path=path, enabled=extension_dirname not in disabled_extensions, is_builtin=is_builtin, metadata=metadata)

            extensions.append(extension)
            loaded_extensions[canonical_name] = extension

    for extension in extensions:
        if not extension.enabled:
            continue

        for req in extension.metadata.requires:
            required_extension = loaded_extensions.get(req)
            if required_extension is None:
                errors.report(f'Extension "{extension.name}" requires "{req}" which is not installed.', exc_info=False)
                continue

            if not required_extension.enabled:
                errors.report(f'Extension "{extension.name}" requires "{required_extension.name}" which is disabled.', exc_info=False)
                continue
