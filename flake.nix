{
  description = "Infernix development environment (CUDA 13.1, Python 3.13)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      # x86_64-linux only: the product targets sm_120a on Linux and Windows, and this
      # harness is the Linux workstation's.
      system = "x86_64-linux";

      # The CUDA toolkit is unfree (CUDA EULA).
      pkgs = (import nixpkgs) {
        inherit system;
        config.allowUnfree = true;
      };

      # CUDA 13.1, matching the validated toolkit (docs/maintainer/build-system.md:4) and
      # the 13.1.2 images the product container pins (Dockerfile).
      cuda = pkgs.cudaPackages_13_1;

      # Python 3.13, mirroring the FrInfer harness this flake is vendored from. The
      # maintained interpreter is 3.11 (docs/maintainer/build-system.md), but at the
      # pinned nixpkgs revision `python311Packages.torch` cannot build at all: its
      # required modules include sphinx 9.1.0, which needs Python >= 3.12. A 3.11 shell
      # without torch would be unable to run the registered torch-importing interop test
      # (tests/artifact/flash_next_interop.py) or the converter suites, so 3.13 it is.
      #
      # The set covers what every registered CTest and the expert-cache replay harness
      # import: numpy (tools/expert_cache_replay/, tools/artifact/), jinja2 and jsonschema
      # (tests/text/, tests/models/qwen3_5/), torch and safetensors (the artifact and
      # converter interop tests), plus pytest, pyyaml and rich for the suites and tooling.
      # It deliberately omits modules used only by standalone, function-local or
      # optional-imported scripts: transformers (tools/bench/ corpus and fixture
      # builders, tools/flash_next/upstream_check.py), sympy, modelopt, gguf, PIL and
      # evalscope. None is reachable from a registered test; add any of them below if
      # such a script is needed.
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
          # <nvml.h> for tools/flash_next_probe/vram_probe.cpp. The probe loads NVML at
          # run time, but its compile still needs the header: without this the
          # BUILD_TESTING=ON tree fails with "fatal error: nvml.h: No such file or directory".
          cuda.cuda_nvml_dev
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
