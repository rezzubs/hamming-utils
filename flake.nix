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
    };

    formatter.${system} = pkgs.alejandra;
  };
}
