#!/usr/bin/env python3
"""
Container entrypoint for PINNGBM Docker image.
Reads from PredictGBM input structure and runs Engine with command-line options.
Usage: python container_entrypoint.py [option_key option_value ...]
"""
import os
import sys
import shutil
from pathlib import Path
from loguru import logger
import nibabel as nib
import numpy as np

# Add current directory to path (container_inference.py is in /app)
# All PINNGBM modules are also in /app
sys.path.insert(0, '/app')

from Engine import Engine
from Options import Options


def setup_predictgbm_paths(opts_obj: Options, input_dir: Path, subject_id: str = "00000") -> None:
    """
    Set up file paths from PredictGBM input structure.
    Expected structure: /mlcube_io0/Patient-00000/00000-{modality}.nii.gz
    """
    patient_dir = input_dir / f"Patient-{subject_id}"
    
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    
    # Map PredictGBM naming to PINNGBM expected paths
    modality_map = {
        'gm_file': f'{subject_id}-gm.nii.gz',
        'wm_file': f'{subject_id}-wm.nii.gz',
        'csf_file': f'{subject_id}-csf.nii.gz',
        'segpre_file': f'{subject_id}-tumorseg.nii.gz',
    }
    
    for opt_key, filename in modality_map.items():
        filepath = patient_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required input file not found: {filepath}")
        
        opts_obj.opts['dataset_opts'][opt_key] = str(filepath)
        logger.debug(f"Set {opt_key} = {filepath}")


def save_nifti_output(u_pred: np.ndarray, output_path: Path, ref_nifti_path: Path) -> None:
    """Save prediction array as NIfTI file using reference geometry."""
    ref_img = nib.load(ref_nifti_path)
    affine = ref_img.affine
    header = ref_img.header
    
    nifti_img = nib.Nifti1Image(u_pred, affine, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, output_path)
    logger.info(f"Saved prediction to {output_path}")


def main():
    """Main entrypoint for Docker container."""
    # PredictGBM I/O paths
    input_dir = Path(os.environ.get("MLCUBE_INPUT_DIR", "/mlcube_io0"))
    output_dir = Path(os.environ.get("MLCUBE_OUTPUT_DIR", "/mlcube_io1"))
    subject_id = os.environ.get("PINNGBM_SUBJECT_ID", "00000")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PINNGBM Container Entrypoint")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Subject ID: {subject_id}")
    
    # Initialize Options
    opts = Options()
    
    # Set device based on environment (can be overridden by command-line args)
    force_cpu = os.environ.get("PINNGBM_FORCE_CPU", "false").lower() in ("true", "1", "yes")
    cuda_device = os.environ.get("CUDA_DEVICE", "0")
    
    if force_cpu:
        opts.opts['device'] = 'cpu'
        logger.info("Forcing CPU execution (from environment)")
    else:
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                opts.opts['device'] = f'cuda:{cuda_device}' if cuda_device else 'cuda'
                logger.info(f"Using CUDA device: {opts.opts['device']}")
            else:
                opts.opts['device'] = 'cpu'
                logger.warning("CUDA not available, falling back to CPU")
        except ImportError:
            opts.opts['device'] = 'cpu'
            logger.warning("PyTorch not available, using CPU")
    
    # Configure logger for container environment (can be overridden by args)
    opts.opts['logger_opts']['use_mlflow'] = False
    opts.opts['logger_opts']['use_stdout'] = True
    opts.opts['logger_opts']['use_tmp'] = True
    opts.opts['skip_fig'] = True
    
    # Enable figures for debugging (set skip_fig=False)
    # opts.opts['flags'] = 'small'
    # opts.opts['dataset_opts']['xdim'] = 2

    
    # Parse command-line arguments (same format as runexp.py)
    # Skip script name, parse remaining args
    if len(sys.argv) > 1:
        logger.info(f"Parsing command-line arguments: {sys.argv[1:]}")
        opts.parse_args(*sys.argv[1:])
    else:
        logger.info("No command-line arguments provided, using defaults")
    
    # Set up input file paths from PredictGBM structure
    # This happens after parsing args so args can override paths if needed
    setup_predictgbm_paths(opts, input_dir, subject_id)
    
    # Process options
    opts.process_options()
    
    logger.info("Running PINNGBM pipeline...")
    
    # Initialize and run Engine
    try:
        engine = Engine(opts.opts)
        engine.run()
        
        # Extract final prediction
        if 'upred_fdm' in engine.vdict:
            upred = engine.vdict['upred_fdm']
        elif 'char_fdm' in engine.vdict:
            upred = engine.vdict['char_fdm']
        else:
            raise RuntimeError("Engine did not produce 'upred_fdm' or 'char_fdm' output. Check pipeline execution.")
        
        # Get final time point
        phiu_final = upred[..., -1]
        
        logger.info(f"Prediction shape: {phiu_final.shape}")
        
        # Save output in PredictGBM format
        output_file = output_dir / f"{subject_id}.nii.gz"
        
        # Use WM file as reference for geometry
        wm_path = Path(opts.opts['dataset_opts']['wm_file'])
        save_nifti_output(phiu_final, output_file, wm_path)
        
        
        logger.info("=" * 60)
        logger.info("PINNGBM pipeline completed successfully!")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 60)
        
        # Close logger
        if engine.logger:
            engine.logger.close()
        
    except Exception as e:
        logger.error(f"PINNGBM execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()