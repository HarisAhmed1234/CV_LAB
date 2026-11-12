# Texture Analysis for Material Classification

## Pipeline
Convert each image to grayscale and resize it, extract texture features, then train a classifier (an SVM here) to separate wood, metal and fabric.

## Technique 1: Local Binary Pattern (LBP)
**How it works:** For each pixel, its P neighbours on a circle of radius R are compared with the centre pixel. A neighbour is 1 if its intensity is greater than or equal to the centre, otherwise 0. These bits form a binary code, and the histogram of codes over the image is the feature vector. The "uniform" variant keeps only patterns with at most two 0/1 transitions, which gives a compact histogram of P + 2 bins.

**Suitable textures:**
- Wood: fine grain lines and knots.
- Fabric: repetitive weave patterns.
- Brushed metal: fine directional micro-texture.

## Technique 2: GLCM (Grey-Level Co-occurrence Matrix) statistics
**How it works:** The GLCM counts how often pairs of grey levels occur at a given distance and angle. Statistics are then taken from it:
- Contrast: local intensity variation.
- Energy: uniformity.
- Homogeneity: closeness of values to the diagonal.
- Correlation: linear dependency between neighbouring pixels.

Averaging over several angles (0°, 45°, 90°, 135°) gives some rotation robustness.

**Suitable textures:**
- Smooth metal: high energy and homogeneity, low contrast.
- Coarse fabric or rough wood: high contrast, low energy.

(The lab's texture energy and contrast histograms from section 3.6 are similar local-statistics features and could be used in the same way.)

## Advantages
- **LBP:** simple and fast. It is invariant to monotonic lighting changes and gives a compact feature vector.
- **GLCM:** interpretable features with direct physical meaning (smooth vs rough, uniform vs varied). It captures spatial relationships between pixels, which plain histograms miss.
- **Combined:** LBP captures micro-patterns and GLCM captures global statistics, so together they are more discriminative than either alone.

## Limitations
- **LBP:** sensitive to noise on near-flat regions, and plain LBP is not rotation invariant. It uses only local information and ignores colour.
- **GLCM:** the choice of distance, angle and grey-level quantisation matters, and it is computationally heavier. Its features can be similar for different materials with similar roughness.
- **Both:** performance drops with changes in scale, strong lighting changes such as metal reflections and specular highlights, and mixed or printed surfaces (for example wood-effect fabric). Shiny metal can look different under different lighting, so a varied dataset or normalisation is needed.
- Colour is ignored. Adding a colour histogram would help separate classes such as metal and wood.
