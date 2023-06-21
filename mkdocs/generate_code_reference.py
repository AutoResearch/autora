"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src_paths = sorted(Path("./temp_dir/").rglob("**/src"))


def source_file_generator(src_paths):
    """Generate the paths and src directory paths of each python file."""
    for src_path in src_paths:
        for path in sorted(src_path.rglob("**/*.py")):
            yield path, src_path


for path, src_path in source_file_generator(src_paths):
    module_path = path.relative_to(src_path).with_suffix("")
    doc_path = path.relative_to(src_path).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        if parts[-1] == "__init__":
            ident = ".".join(parts[:-1])
        else:
            ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
