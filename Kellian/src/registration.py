from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None


def affine_register_target_to_query(query_path: Path, target_path: Path) -> np.ndarray:
    if sitk is None:
        raise RuntimeError("SimpleITK is required for affine registration.")
    fixed = sitk.ReadImage(str(query_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(target_path), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=80,
        gradientMagnitudeTolerance=1e-5,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled).astype(np.float32)
