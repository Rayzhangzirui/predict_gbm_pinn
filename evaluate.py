import numpy as np
from scipy.ndimage import distance_transform_edt

def create_standard_plan(segmentation: np.ndarray, brain_mask: np.ndarray, ctv_margin: int) -> np.ndarray:
    """
    Creates a standard radiotherapy plan by dilating the segmentation and masking with the brain mask.

    Parameters:
        segmentation (np.ndarray): Binary array of tumor core.
        brain_mask (np.ndarray): Binary array of brain mask.
        ctv_margin (int): Margin in pixels/voxels to dilate.

    Returns:
        np.ndarray: Binary mask of the standard plan.
    """
    if ctv_margin < 0:
        raise ValueError("ctv_margin must be non-negative.")
    
    # Distance transform on the background (inverse of segmentation)
    # distance_transform_edt calculates distance to nearest non-zero pixel. 
    # We want distance from tumor (non-zero) into background (zero).
    # So we compute distance on inverted segmentation.
    # Pixels inside tumor have distance 0.
    # Note: segmentation should be boolean or 0/1.
    
    # only necrotic and enhancing region
    region_of_interst = (segmentation == 1) | (segmentation == 3) 
    dist = distance_transform_edt(~region_of_interst)
    
    # Dilate: pixels within ctv_margin distance
    dilated = dist <= ctv_margin
    
    # Apply brain mask: set everything outside of brain mask to be 0
    standard_plan = dilated & (brain_mask > 0)
    
    return standard_plan.astype(np.int32)

def find_threshold(predicted_density: np.ndarray, target_vol: float, max_iter: int = 100, tol: float = 0.01) -> float:
    """
    Finds the threshold for predicted_density using binary search such that 
    the volume of (predicted_density > threshold) is close to target_vol.

    Parameters:
        predicted_density (np.ndarray): Array of predicted tumor density/probability.
        target_vol (float): Desired volume (number of voxels).
        max_iter (int): Maximum iterations for binary search.
        tol (float): Relative tolerance for volume difference.

    Returns:
        float: The found threshold.
    """
    low = 0.001
    high = 1.0
    
    # Check bounds
    vol_low = np.sum(predicted_density > low)
    if vol_low < target_vol:
        return low # Even lowest threshold gives too little volume
        
    vol_high = np.sum(predicted_density > high)
    if vol_high > target_vol:
        return high # Even highest threshold gives too much volume

    prev_vol = -1.0

    for _ in range(max_iter):
        mid = (low + high) / 2
        current_vol = np.sum(predicted_density > mid)
        
        # Check convergence: close to target volume
        if abs(current_vol - target_vol) / (target_vol + 1e-6) < tol:
            return mid
            
        # Check convergence: consecutive change of volume is small
        if prev_vol != -1.0 and abs(current_vol - prev_vol) / (target_vol + 1e-6) < (tol / 10.0):
             return mid

        prev_vol = current_vol

        # Binary search update
        # Volume is monotonically decreasing with increasing threshold
        if current_vol > target_vol:
            # Volume too big -> need higher threshold to reduce volume
            low = mid
        else:
            # Volume too small -> need lower threshold to increase volume
            high = mid
            
        # Check if bounds are too close
        if (high - low) < 1e-6:
            break
            
    return (low + high) / 2

def evaluate_personalized_plan(
    segmentation: np.ndarray, 
    brain_mask: np.ndarray, 
    predicted_density: np.ndarray, 
    recurrence_mask: np.ndarray, 
    ctv_margin: int = 15
) -> dict:
    """
    Evaluates the personalized plan derived from predicted density.
    
    1. Generates standard plan from segmentation.
    2. Calculates target volume from standard plan.
    3. Finds threshold for predicted density to match target volume.
    4. Creates personalized plan.
    5. Computes coverage of recurrence mask.

    Returns:
        dict: Contains coverage metrics and threshold.
    """
    # 1. Standard Plan
    standard_plan = create_standard_plan(segmentation, brain_mask, ctv_margin)
    std_ctv_vol = np.sum(standard_plan)
    
    # 2. Find Threshold
    threshold = find_threshold(predicted_density, std_ctv_vol)
    
    # 3. Personal Plan
    personal_plan = (predicted_density > threshold) & (brain_mask > 0)
    my_ctv_vol = np.sum(personal_plan)
    
    # 4. Compute Coverage
    # Coverage = (Recurrence AND Plan) / Recurrence
    recurrence_vol = np.sum(recurrence_mask > 0)
    
    if recurrence_vol == 0:
        my_ctv_eff = np.nan
        std_ctv_eff = np.nan
    else:
        my_ctv_coverage = np.sum((recurrence_mask > 0) & (personal_plan > 0))
        my_ctv_eff = my_ctv_coverage / recurrence_vol
        std_ctv_coverage = np.sum((recurrence_mask > 0) & (standard_plan > 0))
        std_ctv_eff = std_ctv_coverage / recurrence_vol
        
    metrics = {
        "std_ctv_vol": std_ctv_vol,
        "my_ctv_vol": my_ctv_vol,
        "threshold": threshold,
        "my_ctv_eff": my_ctv_eff,
        "std_ctv_eff": std_ctv_eff
    }

    plans = {"standard_plan": standard_plan, "personal_plan": personal_plan}

    return metrics, plans

def evaluate_prediction(
    predicted_density: np.ndarray,
    segmentation: np.ndarray,
    th_wt: float,
    th_tc: float
) -> dict:
    """
    Evaluates prediction against segmentation using Dice scores.
    
    Args:
        predicted_density: (nx, ny, nz) or (nx, ny)
        segmentation: (nx, ny, nz) or (nx, ny) with labels
        th_wt: threshold for Whole Tumor (Edema + Necrotic + Enhanced)
        th_tc: threshold for Tumor Core (Necrotic + Enhanced)
        
    Returns:
        dict: {'dice_wt': float, 'dice_tc': float}
    """
    # GT Masks
    # WT: 1 (Necrotic), 2 (Edema), 3 (Enhancing)
    gt_wt = np.isin(segmentation, [1, 2, 3])
    # TC: 1 (Necrotic), 3 (Enhancing)
    gt_tc = np.isin(segmentation, [1, 3])
    
    # Pred Masks
    pred_wt = predicted_density > th_wt
    pred_tc = predicted_density > th_tc
    
    def dice(a, b):
        a = np.asarray(a).astype(bool)
        b = np.asarray(b).astype(bool)
        intersection = np.sum(a & b)
        size_a = np.sum(a)
        size_b = np.sum(b)
        if size_a + size_b == 0:
            return 1.0 if intersection == 0 else 0.0
        return 2.0 * intersection / (size_a + size_b)
        
    return {
        'dice_wt': dice(pred_wt, gt_wt),
        'dice_tc': dice(pred_tc, gt_tc)
    }

