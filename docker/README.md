# PINNGBM Docker Container

This directory contains Dockerfiles and build scripts for creating PINNGBM containers compatible with the PredictGBM pipeline.

## Building Containers

### CPU Version (for Mac/local testing)

```bash
cd pinngbmtorch
bash docker/build_cpu.sh
```

This creates:
- Docker image: `pinngbm:cpu`
- Tar file: `pinngbm_cpu.tar` (in project root)

### GPU Version (for Linux cluster)

```bash
cd pinngbmtorch
bash docker/build_gpu.sh
```

This creates:
- Docker image: `pinngbm:latest`
- Tar file: `pinngbm.tar` (in project root)

## Using with PredictGBM

1. Copy the tar file to PredictGBM's models directory:
   ```bash
   cp pinngbm_cpu.tar ../PredictGBM/predict_gbm/data/models/
   ```

2. Use in PredictGBM:
   ```python
   from predict_gbm.prediction.predict import predict_tumor_growth
   
   predict_tumor_growth(
       tumorseg_file=Path("tumor_seg.nii.gz"),
       gm_file=Path("gm_pbmap.nii.gz"),
       wm_file=Path("wm_pbmap.nii.gz"),
       csf_file=Path("csf_pbmap.nii.gz"),
       model_id="pinngbm_cpu",  # or "pinngbm" for GPU version
       outdir=Path("output"),
       cuda_device="0",
   )
   ```

## Testing Locally

See `test_container.sh` and `test_docker_container.sh` in the project root for local testing examples.

## Container Options

The container accepts command-line arguments in the same format as `runexp.py`:

```bash
docker run pinngbm:cpu \
    flags small \
    traintype full \
    pde_opts.factor 10.0 \
    train_opts.iter_init 4000
```

Available options are defined in `Options.py`. Use dot notation for nested options (e.g., `pde_opts.factor`).

## Environment Variables

- `MLCUBE_INPUT_DIR`: Input directory (default: `/mlcube_io0`)
- `MLCUBE_OUTPUT_DIR`: Output directory (default: `/mlcube_io1`)
- `PINNGBM_SUBJECT_ID`: Subject ID (default: `00000`)
- `PINNGBM_FORCE_CPU`: Force CPU execution (`true`/`false`, default: `false`)
- `CUDA_DEVICE`: CUDA device ID (default: `0`)
