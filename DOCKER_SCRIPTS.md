# Docker Build Script

This directory contains a comprehensive script for building and pushing Docker images for Text Embeddings Inference with multiple CUDA compute capabilities.

## Script Overview

### `push-docker-images.sh` (Comprehensive)
Full-featured script with error handling, parallel builds, and verification.

**Features:**
- Multi-registry support (GHCR + Internal HF)
- Parallel builds for faster execution
- Image verification after pushing
- Comprehensive logging and error handling
- Dry-run mode for testing
- Support for all variants including Blackwell (sm120)

**Usage:**
```bash
# Build all variants sequentially
./push-docker-images.sh

# Build in parallel (faster)
./push-docker-images.sh --parallel

# Dry run to see what would be built
./push-docker-images.sh --dry-run

# Verify existing images only
./push-docker-images.sh --verify

# Show help
./push-docker-images.sh --help
```

## Supported Variants

| Prefix | Architecture | Compute Cap | Dockerfile | gRPC Support |
|--------|--------------|-------------|------------|--------------|
| `turing-` | Turing | sm75 | Dockerfile-cuda | ✅ |
| (none) | Ampere | sm80 | Dockerfile-cuda | ✅ |
| `86-` | A10 | sm86 | Dockerfile-cuda | ✅ |
| `89-` | RTX 4000 | sm89 | Dockerfile-cuda | ✅ |
| `hopper-` | Hopper | sm90 | Dockerfile-cuda | ✅ |
| `blackwell-` | Blackwell | sm120 | Dockerfile-cuda | ✅ |

## Image Naming Convention

Images are tagged following the GitHub Actions conventions:

**Standard Images:**
- `{prefix}latest` - Latest version
- `{prefix}1.8.4` - Full version
- `{prefix}1.8` - Major.minor version

**gRPC Images:**
- `{prefix}latest-grpc` - Latest gRPC version
- `{prefix}1.8.4-grpc` - Full gRPC version
- `{prefix}1.8-grpc` - Major.minor gRPC version

**Examples:**
- `ghcr.io/huggingface/text-embeddings-inference:latest` (sm80)
- `ghcr.io/huggingface/text-embeddings-inference:turing-1.8.4` (sm75)
- `ghcr.io/huggingface/text-embeddings-inference:blackwell-1.8.4-grpc` (sm120)

## Registries

**Primary Registry:**
- `ghcr.io/huggingface/text-embeddings-inference`

**Internal Registry (if available):**
- `registry.internal.huggingface.tech/api-inference/text-embeddings-inference`

## Prerequisites

1. **Docker & Docker Buildx:**
   ```bash
   docker --version
   docker buildx version
   ```

2. **Registry Access:**
   ```bash
   docker login ghcr.io
   # For internal registry:
   docker login registry.internal.huggingface.tech
   ```

3. **Git Repository:**
   - Scripts must be run from the repository root
   - Clean git state recommended

## Environment Variables

Optional environment variables for better performance:

```bash
export DOCKER_BUILDKIT=1
export BUILDKIT_INLINE_CACHE=1
```

## Build Configuration

**Common Build Args:**
- `CUDA_COMPUTE_CAP` - Target compute capability
- `GIT_SHA` - Git commit SHA
- `SCCACHE_GHA_ENABLED` - Enable sccache caching
- `DEFAULT_USE_FLASH_ATTENTION=False` - For Turing (sm75) only

**Platform:**
- `linux/amd64` - All images built for this platform
- Multi-platform support can be added if needed

## Troubleshooting

### Common Issues

1. **Permission Denied:**
   ```bash
   chmod +x push-docker-images.sh quick-docker-build.sh
   ```

2. **Registry Login:**
   ```bash
   docker logout ghcr.io
   docker login ghcr.io
   ```

3. **Buildx Not Available:**
   ```bash
   docker buildx install
   docker buildx create --use
   ```

4. **Out of Memory:**
   - Reduce parallel builds
   - Use sequential mode: `./push-docker-images.sh`

### Debug Mode

Enable verbose logging:
```bash
export DOCKER_BUILDKIT=1
docker buildx build --progress=plain ...
```

## Version Information

- **Current Version:** 1.8.4
- **Git SHA:** `git rev-parse --short HEAD`
- **Matrix Config:** `.github/workflows/matrix.json`

## Integration with CI/CD

These scripts follow the same patterns as the GitHub Actions workflow in `.github/workflows/build.yaml`. They can be used for:

- Local testing before CI runs
- Manual image builds for specific variants
- Emergency deployments
- Development and debugging

## Performance Tips

1. **Use sccache:** Enabled by default for faster compilation
2. **Parallel builds:** Use `--parallel` flag for multiple variants
3. **Build caching:** Inline cache enabled by default
4. **Selective builds:** Build only needed variants with quick script

## Security Notes

- Images are built with security scanning in CI
- No secrets are embedded in images
- Minimal base images used
- Regular security updates applied