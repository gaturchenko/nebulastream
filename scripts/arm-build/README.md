# Building the Raspberry Pi binaries on GCP Batch

Builds `systest` and `nes-single-node-worker` for aarch64 on a GCP Batch VM, and leaves a tarball in Cloud Storage that can be
copied to the Pi.

```
scripts/arm-build/
├── submit_arm_build.sh    # packages the working tree, renders the job, prints the submit command
├── batch-job.json.tmpl    # the Batch job: n4a-standard-16, SPOT, Cloud Logging (your smoke test, with a bigger disk)
└── build_nes_arm.sh       # what runs on the VM: toolchain, dependencies, build, package, upload
```

`submit_arm_build.sh` never submits anything unless you pass `--submit`; by default it only prints the commands.

## One-time setup

**1. Egress.** The build downloads the LLVM toolchain, the vcpkg dependency sources, crates and the ANTLR jar, so the VM needs
internet access. The project has no Cloud NAT, and `--no-external-ip-address` leaves the VM without egress, so create one:

```bash
gcloud compute routers create arm-build-router \
    --network=arm-build-vpc --region=europe-west3

gcloud compute routers nats create arm-build-nat \
    --router=arm-build-router --region=europe-west3 \
    --auto-allocate-nat-external-ip-addresses --nat-all-subnet-ip-ranges
```

**2. Bucket** for the sources, the dependency cache and the artifacts:

```bash
gcloud storage buckets create gs://dams-nes-arm-builds --location=europe-west3

# Keep Cloud Storage traffic inside the VPC
gcloud compute networks subnets update arm-build-subnet \
    --region=europe-west3 --enable-private-ip-google-access
```

**3. Permissions.** The VM runs as the default Compute Engine service account, which must be able to read and write that bucket:

```bash
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding gs://dams-nes-arm-builds \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/storage.objectAdmin
```

## Running a build

```bash
# Packages the current working tree (uncommitted changes included) and prints what to run
scripts/arm-build/submit_arm_build.sh --bucket gs://dams-nes-arm-builds

# ... or do the upload and submission in one go
scripts/arm-build/submit_arm_build.sh --bucket gs://dams-nes-arm-builds --submit
```

Useful flags: `--targets "systest"`, `--machine-type`, `--disk-gb`, `--max-run-duration`, `--provisioning-model STANDARD`,
`--job-name`, `--location`.

Watch it, then collect the artifact:

```bash
gcloud beta batch jobs tail JOB_NAME --location=europe-west3
gcloud storage cp 'gs://dams-nes-arm-builds/arm-build/STAMP/nes-arm64-*.tar.zst' .
```

## What the job does

1. Installs the toolchain the project's own images use: clang-19 from apt.llvm.org, CMake 3.31 from Kitware, ninja, mold, Rust
   (for the `nes-network` bindings through Corrosion) and a JRE (for the ANTLR jar).
2. Downloads the prebuilt `nes-llvm-…-arm64-none-local` toolchain that Nautilus' MLIR backend compiles against, the same one
   `docker/dependency/Dependency.dockerfile` uses.
3. Builds the vcpkg dependencies from this tree's manifest with the `arm64-linux-none-local` triplet. This fork carries its own
   OpenVINO, IREE, libpng and Nautilus ports, so the upstream prebuilt dependency image does not apply.
4. Configures and builds the requested targets in Release.
5. Packages the binaries with a `BUILD_INFO.txt` (revision, compiler, and the highest glibc/libstdc++ symbol versions the
   binaries need) and uploads everything.

**Timing.** The first run is dominated by the dependencies and takes a few hours on 16 vCPUs. The vcpkg binary cache is synced to
`gs://BUCKET/vcpkg-cache/` right after they are built, so later runs for the same manifest skip straight to compiling
NebulaStream. Since the VM is a Spot instance, it can be reclaimed mid-build; the task retries, and the cache means a retry does
not start from zero. Use `--provisioning-model STANDARD` if a long first build keeps getting preempted.

## Deploying to the Pi

```bash
scp nes-arm64-*.tar.zst pi@raspberrypi:/tmp/
ssh pi@raspberrypi 'mkdir -p ~/nes && tar --zstd -xf /tmp/nes-arm64-*.tar.zst -C ~/nes && cat ~/nes/BUILD_INFO.txt'
ssh pi@raspberrypi '~/nes/bin/systest --help | head -3'
```

The binaries are self-contained apart from the system C and C++ runtimes: the vcpkg triplet links the dependencies statically.
Inference systests additionally need `ovc` 2025.3 on the Pi's `PATH` for model import, as on the development machine.

## If the job fails early

The container logs are in Cloud Logging (`gcloud beta batch jobs tail JOB --location=europe-west3`). On any failure the script
prints vcpkg's `issue_body.md` — which embeds the failing port's own build log — into the job log, and uploads the rest
(`vcpkg-manifest-install.log`, the buildtree logs, CMake's configure log) to
`gs://BUCKET/arm-build/STAMP/diagnostics-<timestamp>.tar.gz`. The VM's disk is discarded the moment the task ends, so those two
are the only record of what went wrong.

The checks that fail fast, before any expensive work:

* **`exit status 8` right after "Fetching the prebuilt LLVM/MLIR toolchain"** — `wget` got an HTTP error, meaning the asset name
  does not exist under that release tag. List what is published and set `NES_LLVM_VERSION` / `NES_LLVM_STDLIB` in
  `build_nes_arm.sh` accordingly:

  ```bash
  curl -s https://api.github.com/repos/nebulastream/clang-binaries/releases/tags/vmlir-21-with-fix-173075 \
    | grep -o '"name": "nes-llvm[^"]*"'
  ```

  Note that the asset's stdlib token is the standard library's own name (`libstdcxx`), while the vcpkg triplet calls the same
  choice `local`.
* **`CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS-NOTFOUND: not found`, usually surfacing as `Could NOT find Threads` in a Boost port** —
  `clang-scan-deps` is missing. The chainload toolchain compiles as C++23, so CMake scans for modules in `try_compile`, and
  without the scanner every such check fails. It comes from the `clang-tools-<version>` package.
* **`vcpkg was unable to detect the active compiler's information`, with a `…-libcxx` triplet in the path** — the build picked
  libc++ instead of the system standard library. `cmake/ImportDependencies.cmake` derives the triplet from
  `USE_LIBCXX_IF_AVAILABLE` and ignores a `VCPKG_TARGET_TRIPLET` given on the command line, so that option has to be `OFF`
  (which `build_nes_arm.sh` passes). The Pi has no libc++ runtime, and the prebuilt LLVM fetched above is the libstdc++ build.
* **`vcpkg was cloned as a shallow repository`** — vcpkg resolves the manifest baseline and the port trees through git history,
  so the clone in `build_nes_arm.sh` must not use `--depth`. The follow-up `CMake was unable to find a build program
  corresponding to "Ninja"` in the same output is only fallout: CMake gives up inside `project()` before resolving the generator,
  even though ninja is installed.
* **A failure during `apt-get update`** — the VM has no egress; check that the Cloud NAT exists for `arm-build-vpc` in
  `europe-west3`.
* **`AccessDeniedException` on a `gs://` path** — the Compute Engine service account lacks `roles/storage.objectAdmin` on the
  bucket.

After fixing the script, re-upload just it to the same prefix and submit again under a new job name; the source tarball and job
definition already in the bucket stay valid:

```bash
gcloud storage cp scripts/arm-build/build_nes_arm.sh gs://BUCKET/arm-build/STAMP/build_nes_arm.sh
gcloud batch jobs submit JOB-retry --location=europe-west3 --config=/tmp/nes-arm-build-STAMP.json \
    --network=... --subnetwork=... --no-external-ip-address
```

## Why these choices

* **Ubuntu 24.04 container, not the Batch VM image.** The Pi runs Raspberry Pi OS Trixie (glibc 2.41, libstdc++ from GCC 14).
  A binary may use an older glibc than its host but never a newer one, so the builder's glibc has to be no newer than the Pi's.
  Ubuntu 24.04 gives glibc 2.39 plus the GCC 14 standard library the C++23 code needs, and matches `docker/dependency/Base.dockerfile`.
  `BUILD_INFO.txt` records the actual symbol versions required, so a mismatch is visible before copying anything to the Pi.
* **No `-march=native`.** The builder is an Axion (ARMv8.2+) and the Pi is ARMv8.0-A. `CMakeLists.txt` pins `-march=armv8-a` for
  arm64 and `vcpkg/custom-triplets/bits/arch/arm64.cmake` does the same for the dependencies, so nothing emits instructions the
  Pi lacks. Leave `NES_BUILD_NATIVE` off.
* **`USE_LOCAL_MLIR=ON`.** It makes the build use the prebuilt LLVM/MLIR archive. Without it the project enables the manifest's
  `mlir` feature, which pulls in the `llvm` port and builds LLVM 21.1.8 from source — hours of compute, and stock LLVM rather
  than the patched `21-with-fix-173075` the x86 builds use.
* **`NES_ENABLES_TESTS=ON`.** `systest` is registered through `add_tests_if_enabled`, so it is not built otherwise.
* **`USE_LIBCXX_IF_AVAILABLE=OFF`.** It selects the system standard library, which makes the vcpkg triplet
  `arm64-linux-none-local` and matches the `-libstdcxx` flavour of the prebuilt LLVM. Raspberry Pi OS ships libstdc++ but not
  libc++, so this is also what the Pi can actually run.
