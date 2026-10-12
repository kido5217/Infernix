{
  description = "Infernix development environment (CUDA 13.1, Python 3.13)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # The CUDA toolkit is unfree (CUDA EULA).
      pkgs = (import nixpkgs) {
        inherit system;
        config.allowUnfree = true;
      };

      cuda = pkgs.cudaPackages_13_1;

      # Python 3.13, mirroring the FrInfer harness this flake is vendored from. The
      # maintained interpreter is 3.11 (docs/maintainer/build-system.md), but at the
      # pinned nixpkgs revision `python311Packages.torch` cannot build at all: its
      # required modules include sphinx 9.1.0, which needs Python >= 3.12. A 3.11 shell
      # without torch would be unable to run the registered torch-importing interop test
      # (tests/artifact/flash_next_interop.py) or the converter suites, so 3.13 it is.
      # The set is what the tree's Python imports: numpy (expert-cache replay, artifact
      # tools), jinja2/jsonschema (the registered text tests under tests/text/),
      # torch/safetensors (artifact and converter interop), pytest, pyyaml, rich.
      # Not included: `transformers`, imported only by the bench corpus/fixture builders
      # (tools/bench/) and tools/flash_next/upstream_check.py; add `ps.transformers`
      # here if a bench fixture build is needed.
      python = pkgs.python313.withPackages (
        ps: with ps; [
          jinja2
          jsonschema
          numpy
          pytest
          pyyaml
          rich
          safetensors
          torch
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "infernix-dev";
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_profiler_api
          cuda.cuda_nvtx
          cuda.nsight_compute
          cuda.nsight_systems
          pkgs.ffmpeg
          pkgs.curl
          python
        ];
        shellHook = ''
          # NixOS exposes the proprietary NVIDIA driver libraries, including libcuda.so.1,
          # under /run/opengl-driver/lib. Without that directory on LD_LIBRARY_PATH the
          # CUDA runtime cannot find the driver and reports the misleading
          # "CUDA driver version is insufficient for CUDA runtime version". Prepend it
          # once, leaving any existing LD_LIBRARY_PATH entries in place.
          case ":''${LD_LIBRARY_PATH:-}:" in
            *:/run/opengl-driver/lib:*) ;;
            *) export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
          esac
        '';
      };
    };
}
