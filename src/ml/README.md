# Machine Learning Package

Provides machine learning implementations using quaternion-based neural networks and related algorithms.

## Components

- **QuaternionPerceptron**: Core quaternion-based perceptron implementation
- **LogisticRegression**: Placeholder for future logistic regression implementation
- **quaterion.py**: Python reference implementation for comparison

## QuaternionPerceptron

A perceptron model that uses quaternions for weights, inputs, and outputs.

### Architecture

This implementation uses two quaternion weights, representing rotations, that operate in different coordinate frames:

- **Bias rotation** - Applied first as world-frame rotation: `biasRotation * inputOrientation * actionRotation`
- **Action rotation** - Applied last as local-frame rotation: `biasRotation * inputOrientation * actionRotation`

The quaternion chain allows the model to learn both global transformations and input-specific adjustments through separate rotation parameters.

### Key Features

- **Quaternion-based weights**: Uses rotations instead of scalar weights
- **Geodesic learning**: Updates weights using exponential map in Lie algebra
- **Binary classification**: Converts quaternion outputs to binary labels
- **Batch training**: Supports training on multiple samples simultaneously
- **Gradient decomposition**: Decomposes error rotations into bias and action updates

### Training Process

1. **Forward Pass**: Applies quaternion chain to input orientation
2. **Error Computation**: Calculates geodesic distance to target orientation
3. **Gradient Decomposition**: Decomposes error rotation into bias and action components
4. **Weight Updates**: Applies gradients using exponential map and quaternion multiplication

### Usage Examples

```java
// Create perceptron
QuaternionPerceptron perceptron = new QuaternionPerceptron(0.01, 42L);

// Single prediction
Quaternion input = new Quaternion(0.0, 1.0, 0.0, 0.5).normalize();
Quaternion output = perceptron.forward(input);

// Classification
int label = perceptron.classify(input);

// Training
List<Quaternion> inputs = Arrays.asList(input);
List<Integer> labels = Arrays.asList(1);
perceptron.step(inputs, labels);
```

### Mathematical Foundation

The perceptron operates in the Lie group SO(3) represented by unit quaternions:

- **Weights**: Represent rotations in 3D space
- **Inputs**: Represent orientations in 3D space
- **Updates**: Use exponential map to sum rotation vectors in tangent space
- **Distance**: Geodesic distance on the 4D unit sphere (S³)

### Error Handling

- **Singular matrices**: Gracefully handles linearly dependent basis vectors
- **Unit quaternions**: Ensures all quaternions remain normalized
- **Numerical stability**: Robust handling of edge cases and extreme values

## Python Reference Implementation

The `quaterion.py` file contains a Python reference implementation for comparison and validation of the Java implementation.

## Applications

- **3D orientation learning**: Learning spatial relationships and rotations
- **Robotics**: Control systems requiring quaternion representations
- **Computer vision**: 3D pose estimation and tracking
- **Physics simulations**: Systems with rotational dynamics
- **Neural network research**: Exploring geometric learning approaches

## Future Development

- **Octonion extension**: Support for 8D hypercomplex numbers
- **Multi-layer networks**: Deeper quaternion-based architectures
- **Alternative activation functions**: Non-associative operations
- **GPU acceleration**: Parallel quaternion operations
- **Additional loss functions**: Beyond geodesic distance metrics

## License

**Copyright (c) 2025 Hieronim Kubica**

This package is part of V3J, licensed under the MIT License. See the main [LICENSE](../../LICENSE) file for details.

