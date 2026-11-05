#!/usr/bin/env python3
"""
Packaging script for Alforja.
Builds:
  1. RPM package (using rpmbuild)
  2. DEB package (using ar, tar, xz or dpkg-deb)
Outputs are placed into the `dist/` directory.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

VERSION = "0.1.0"
PKG_NAME = "alforja"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def clean_dirs():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def build_deb():
    print(f"\n📦 Building DEB package ({PKG_NAME}_{VERSION}_all.deb)...")
    deb_root = BUILD_DIR / "deb_pkg"
    deb_root.mkdir(parents=True, exist_ok=True)

    # 1. Staging tree
    usr_bin = deb_root / "usr" / "bin"
    usr_share = deb_root / "usr" / "share" / PKG_NAME
    app_dir = deb_root / "usr" / "share" / "applications"
    meta_dir = deb_root / "usr" / "share" / "metainfo"
    icon_dir = deb_root / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"
    doc_dir = deb_root / "usr" / "share" / "doc" / PKG_NAME
    debian_dir = deb_root / "DEBIAN"

    for d in [usr_bin, usr_share, app_dir, meta_dir, icon_dir, doc_dir, debian_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    shutil.copy2(PROJECT_ROOT / "main.py", usr_share / "main.py")
    (usr_share / "main.py").chmod(0o755)

    # Symlink in /usr/bin/alforja
    os.symlink(f"/usr/share/{PKG_NAME}/main.py", usr_bin / "alforja")

    # Data files
    shutil.copy2(
        PROJECT_ROOT / "data" / "io.github.alforja.Alforja.desktop",
        app_dir / "io.github.alforja.Alforja.desktop",
    )
    shutil.copy2(
        PROJECT_ROOT / "data" / "io.github.alforja.Alforja.metainfo.xml",
        meta_dir / "io.github.alforja.Alforja.metainfo.xml",
    )
    shutil.copy2(
        PROJECT_ROOT / "data" / "icons" / "hicolor" / "scalable" / "apps" / "io.github.alforja.Alforja.svg",
        icon_dir / "io.github.alforja.Alforja.svg",
    )
    if (PROJECT_ROOT / "README.md").exists():
        shutil.copy2(PROJECT_ROOT / "README.md", doc_dir / "README.md")
    if (PROJECT_ROOT / "LICENSE").exists():
        shutil.copy2(PROJECT_ROOT / "LICENSE", doc_dir / "copyright")

    # Control file
    shutil.copy2(
        PROJECT_ROOT / "packaging" / "deb" / "DEBIAN" / "control",
        debian_dir / "control",
    )

    deb_output = DIST_DIR / f"{PKG_NAME}_{VERSION}_all.deb"

    # If dpkg-deb is available, use it
    if shutil.which("dpkg-deb"):
        cmd = ["dpkg-deb", "--build", "--root-owner-group", str(deb_root), str(deb_output)]
        subprocess.run(cmd, check=True)
    else:
        # Build manually using debian-binary, control.tar.xz, and data.tar.xz
        staging_tmp = BUILD_DIR / "deb_staging"
        staging_tmp.mkdir(parents=True, exist_ok=True)

        # 1. debian-binary
        with open(staging_tmp / "debian-binary", "wb") as f:
            f.write(b"2.0\n")

        # 2. control.tar.xz
        subprocess.run(
            ["tar", "-cJf", str(staging_tmp / "control.tar.xz"), "--numeric-owner", "--owner=0", "--group=0", "-C", str(debian_dir), "."],
            check=True,
        )

        # 3. data.tar.xz
        subprocess.run(
            ["tar", "-cJf", str(staging_tmp / "data.tar.xz"), "--numeric-owner", "--owner=0", "--group=0", "-C", str(deb_root), "usr"],
            check=True,
        )

        # 4. ar archive
        subprocess.run(
            ["ar", "rcs", str(deb_output), str(staging_tmp / "debian-binary"), str(staging_tmp / "control.tar.xz"), str(staging_tmp / "data.tar.xz")],
            check=True,
        )

    print(f"✅ Created DEB: {deb_output}")


def build_rpm():
    if not shutil.which("rpmbuild"):
        print("⚠️  rpmbuild not found, skipping RPM generation.")
        return

    print(f"\n📦 Building RPM package ({PKG_NAME}-{VERSION}-1.noarch.rpm)...")
    rpm_root = BUILD_DIR / "rpmbuild"
    for sub in ["SOURCES", "SPECS", "BUILD", "RPMS", "SRPMS", "BUILDROOT"]:
        (rpm_root / sub).mkdir(parents=True, exist_ok=True)

    # Create source tarball alforja-0.1.0.tar.gz
    tar_name = f"{PKG_NAME}-{VERSION}.tar.gz"
    tar_path = rpm_root / "SOURCES" / tar_name
    subprocess.run(
        ["tar", "-czf", str(tar_path), f"--transform=s,^,{PKG_NAME}-{VERSION}/,",
         "--exclude=.git", "--exclude=build", "--exclude=dist", "--exclude=__pycache__",
         "-C", str(PROJECT_ROOT), "."],
        check=True,
    )

    spec_path = PROJECT_ROOT / "packaging" / "rpm" / "alforja.spec"
    shutil.copy2(spec_path, rpm_root / "SPECS" / "alforja.spec")

    cmd = [
        "rpmbuild",
        "-ba",
        str(rpm_root / "SPECS" / "alforja.spec"),
        "--define", f"_topdir {rpm_root}",
    ]
    subprocess.run(cmd, check=True)

    # Copy resulting RPMs to dist/
    for rpm_file in (rpm_root / "RPMS").glob("**/*.rpm"):
        shutil.copy2(rpm_file, DIST_DIR / rpm_file.name)
        print(f"✅ Created RPM: {DIST_DIR / rpm_file.name}")


def main():
    clean_dirs()
    build_deb()
    build_rpm()
    print("\n🎉 Packaging complete! Artifacts in dist/:")
    for f in DIST_DIR.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

