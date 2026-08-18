{
  description = "FaultForge development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};

    # uv-managed Python builds are dynamically linked against paths that
    # don't exist on NixOS (no FHS, no dynamic linker at /lib64/...), so
    # they fail to run. Use nixpkgs' Python instead, which is built and
    # patched to work on NixOS, and point uv at it.
    python = pkgs.python314;

    # Libraries that prebuilt (non-nix-built) binaries need but won't find
    # on NixOS, e.g. libstdc++.so.6 for torch/numpy wheels installed by uv.
    # Used for both LD_LIBRARY_PATH (already-running nix-native processes,
    # e.g. Python dlopen()-ing a wheel's compiled extension) and
    # NIX_LD_LIBRARY_PATH (nix-ld resolving a prebuilt binary's own
    # dependencies at startup, e.g. uv-installed ruff/ty). Same libraries
    # are needed in both cases, so both variables share this one list;
    # NIX_LD_LIBRARY_PATH only has any effect if `programs.nix-ld.enable`
    # is set in your NixOS system configuration - harmless otherwise.
    foreignLibraryPath = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib];
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        # Rust
        pkgs.cargo
        pkgs.rustc
        pkgs.clippy
        pkgs.rustfmt
        pkgs.rust-analyzer
        pkgs.cargo-nextest

        # Python
        python
        pkgs.uv
      ];

      # Attributes not recognized by mkShell (packages, shellHook, etc.)
      # are exported as environment variables in the shell.
      RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
      UV_PYTHON_DOWNLOADS = "never";
      UV_PYTHON = "${python}/bin/python3.14";

      LD_LIBRARY_PATH = foreignLibraryPath;
      NIX_LD_LIBRARY_PATH = foreignLibraryPath;
    };

    formatter.${system} = pkgs.alejandra;
  };
}
