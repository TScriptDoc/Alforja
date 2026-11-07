Name:           alforja
Version:        0.1.0
Release:        1%{?dist}
Summary:        Native GTK4/Libadwaita graphical manager for llama.cpp

License:        MIT
URL:            https://github.com/TScriptDoc/Alforja
Source0:        %{name}-%{version}.tar.gz
BuildArch:      noarch

Requires:       python3
Requires:       python3-gobject
Requires:       gtk4
Requires:       libadwaita
Requires:       util-linux

%description
Alforja (Andean saddlebag) is a native Linux desktop application built with
Python, GTK4, and Libadwaita to manage and configure local llama.cpp servers.
It features network sandboxing via unshare, granular process lifecycle control
(including SIGSTOP pause to freeze VRAM state), hardware-aware performance
presets for 4GB GPUs (Nvidia GTX 1650), and an integrated model downloader.

%prep
%autosetup

%build
# Pure python application - no compilation step required

%install
rm -rf %{buildroot}

# Install executable wrapper
install -d %{buildroot}%{_bindir}
install -d %{buildroot}%{_datadir}/%{name}
install -m 0755 main.py %{buildroot}%{_datadir}/%{name}/main.py
ln -s %{_datadir}/%{name}/main.py %{buildroot}%{_bindir}/alforja

# Desktop entry
install -d %{buildroot}%{_datadir}/applications
install -m 0644 data/io.github.alforja.Alforja.desktop %{buildroot}%{_datadir}/applications/

# Appstream Metainfo
install -d %{buildroot}%{_datadir}/metainfo
install -m 0644 data/io.github.alforja.Alforja.metainfo.xml %{buildroot}%{_datadir}/metainfo/

# Scalable Icon
install -d %{buildroot}%{_datadir}/icons/hicolor/scalable/apps
install -m 0644 data/icons/hicolor/scalable/apps/io.github.alforja.Alforja.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/

%files
%license LICENSE
%doc README.md
%{_bindir}/alforja
%{_datadir}/%{name}/
%{_datadir}/applications/io.github.alforja.Alforja.desktop
%{_datadir}/metainfo/io.github.alforja.Alforja.metainfo.xml
%{_datadir}/icons/hicolor/scalable/apps/io.github.alforja.Alforja.svg

%changelog
* Wed Nov 05 2026 Tiago <tiago@local> - 0.1.0-1
- Initial release of Alforja

