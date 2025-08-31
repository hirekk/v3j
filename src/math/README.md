# Math Package

Provides mathematical abstractions for quaternion & octonion algebra and operations.

## Components

- **Quaternion**: Immutable quaternion implementation with comprehensive mathematical operations
- **Octonion**: Placeholder for future octonion implementation

## Quaternion Class

Represents quaternions in the form w + xi + yj + zk where w, x, y, z are real numbers.

### Basic Operations

- **Arithmetic**: Addition, subtraction, multiplication (quaternion and scalar), division
- **Unary Operations**: Negation, conjugation, normalization, inverse
- **Properties**: Norm, norm squared, dot product, cross product

### Advanced Functions

- **Exponential/Logarithm**: `exp()`, `log()`
- **Power**: `pow(double)` for scalar exponents
- **Rotation**: `toRotationMatrix()` for 3D rotation matrices

### Construction Methods

- **Axis-Angle**: `fromAxisAngle(angle, axis)` for rotation quaternions
- **Euler Angles**: `fromEuler(roll, pitch, yaw)` for orientation quaternions
- **Interpolation**: `slerp(q1, q2, t)` for smooth quaternion interpolation

### Constants

- `ZERO`, `ONE`, `I`, `J`, `K` - Standard quaternion basis elements

## Usage Examples

```java
// Create quaternions
Quaternion q1 = new Quaternion(1.0, 2.0, 3.0, 4.0);
Quaternion rotation = Quaternion.fromAxisAngle(Math.PI/2, new double[]{1.0, 0.0, 0.0});

// Perform operations
Quaternion sum = q1.add(rotation);
Quaternion product = q1.multiply(rotation);
Quaternion normalized = q1.normalize();

// Check properties
boolean isUnit = q1.isUnit();
double norm = q1.norm();
```

## Applications

- 3D graphics and game development
- Robotics and control systems
- Computer vision and image processing
- Physics simulations
- Neural network architectures
