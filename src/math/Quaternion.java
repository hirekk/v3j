/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package math;

import java.util.Objects;

/**
 * Represents a quaternion in the form w + xi + yj + zk where w, x, y, z are real numbers.
 *
 * <p>Quaternions extend complex numbers and are useful for representing 3D rotations and
 * orientations. This class provides comprehensive mathematical operations including arithmetic,
 * exponential functions, rotation conversions, and interpolation methods.
 *
 * <p>The class is immutable and thread-safe. All operations return new Quaternion instances.
 *
 * <p><strong>Mathematical Foundation:</strong> Quaternions form a Lie group (SO(3)) with a
 * corresponding Lie algebra (so(3)). The exponential map exp: so(3) → SO(3) and logarithm map log:
 * SO(3) → so(3) provide the connection between the tangent space and the group.
 *
 * <p>Mathematical operations follow standard quaternion algebra rules:
 *
 * <ul>
 *   <li>i² = j² = k² = ijk = -1
 *   <li>ij = k, ji = -k
 *   <li>jk = i, kj = -i
 *   <li>ki = j, ik = -j
 * </ul>
 *
 * <p><strong>Lie Group Operations:</strong> For unit quaternions (SO(3)):
 *
 * <ul>
 *   <li><code>exp()</code>: Maps from tangent space (so(3)) to quaternion group (SO(3))
 *   <li><code>log()</code>: Maps from quaternion group (SO(3)) to tangent space (so(3))
 *   <li><code>log().getVector()</code>: Extracts the rotation vector in tangent space
 * </ul>
 *
 * <p><strong>Example Usage:</strong>
 *
 * <pre>{@code
 * // Create a rotation quaternion (90 degrees around Z-axis)
 * Quaternion rotation = Quaternion.fromAxisAngle(Math.PI / 2, new double[] {0, 0, 1});
 *
 * // Map to tangent space (Lie algebra)
 * Quaternion logRotation = rotation.log();
 * double[] rotationVector = logRotation.getVector(); // [0, 0, π/2]
 *
 * // Map back to quaternion group
 * Quaternion reconstructed = logRotation.exp(); // Should equal rotation
 * }</pre>
 */
public final class Quaternion {
  /** The scalar (real) component of the quaternion */
  private final double w;

  /** The i-component (x-axis) of the quaternion */
  private final double x;

  /** The j-component (y-axis) of the quaternion */
  private final double y;

  /** The k-component (z-axis) of the quaternion */
  private final double z;

  /** Zero quaternion: 0 + 0i + 0j + 0k */
  public static final Quaternion ZERO = new Quaternion(0.0, 0.0, 0.0, 0.0);

  /** Unit quaternion: 1 + 0i + 0j + 0k */
  public static final Quaternion ONE = new Quaternion(1.0, 0.0, 0.0, 0.0);

  /** Unit vector quaternion: 0 + 1i + 0j + 0k */
  public static final Quaternion I = new Quaternion(0.0, 1.0, 0.0, 0.0);

  /** Unit vector quaternion: 0 + 0i + 1j + 0k */
  public static final Quaternion J = new Quaternion(0.0, 0.0, 1.0, 0.0);

  /** Unit vector quaternion: 0 + 0i + 0j + 1k */
  public static final Quaternion K = new Quaternion(0.0, 0.0, 0.0, 1.0);

  /** Numerical stability threshold */
  private static final double NUMERICAL_STABILITY_THRESHOLD = 1e-10;

  /**
   * Constructs a quaternion with the specified components.
   *
   * @param w the scalar component
   * @param x the i-component
   * @param y the j-component
   * @param z the k-component
   * @throws IllegalArgumentException if any component is NaN or infinite
   */
  public Quaternion(double w, double x, double y, double z) {
    if (!Double.isFinite(w) || !Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(z)) {
      throw new IllegalArgumentException("All quaternion components must be finite numbers");
    }
    this.w = w;
    this.x = x;
    this.y = y;
    this.z = z;
  }

  /**
   * Constructs a quaternion from a scalar and a 3D vector.
   *
   * @param w the scalar component
   * @param vector the 3D vector components [x, y, z]
   * @throws IllegalArgumentException if vector does not have exactly 3 components or contains
   *     non-finite values
   */
  public Quaternion(double w, double[] vector) {
    if (vector == null) {
      throw new IllegalArgumentException("Vector cannot be null");
    }
    if (vector.length != 3) {
      throw new IllegalArgumentException("Vector must have exactly 3 components");
    }
    if (!Double.isFinite(w)
        || !Double.isFinite(vector[0])
        || !Double.isFinite(vector[1])
        || !Double.isFinite(vector[2])) {
      throw new IllegalArgumentException("All quaternion components must be finite numbers");
    }
    this.w = w;
    this.x = vector[0];
    this.y = vector[1];
    this.z = vector[2];
  }

  /**
   * Returns the scalar (real) component of this quaternion.
   *
   * @return the w component
   */
  public double getW() {
    return w;
  }

  /**
   * Returns the i-component of this quaternion.
   *
   * @return the x component
   */
  public double getX() {
    return x;
  }

  /**
   * Returns the j-component of this quaternion.
   *
   * @return the y component
   */
  public double getY() {
    return y;
  }

  /**
   * Returns the k-component of this quaternion.
   *
   * @return the z component
   */
  public double getZ() {
    return z;
  }

  /**
   * Returns the vector part of this quaternion as a 3D array.
   *
   * @return array containing [x, y, z] components
   */
  public double[] getVector() {
    return new double[] {x, y, z};
  }

  /**
   * Returns the scalar part of this quaternion.
   *
   * @return the w component (same as getW())
   */
  public double getScalar() {
    return w;
  }

  /**
   * Checks if this quaternion is purely scalar (vector part is zero).
   *
   * @return true if x = y = z = 0, false otherwise
   */
  public boolean isScalar() {
    return x == 0.0 && y == 0.0 && z == 0.0;
  }

  /**
   * Checks if this quaternion is purely vector (scalar part is zero).
   *
   * @return true if w = 0, false otherwise
   */
  public boolean isVector() {
    return w == 0.0;
  }

  /**
   * Checks if this quaternion has unit norm (norm = 1).
   *
   * @return true if the quaternion is normalized, false otherwise
   */
  public boolean isUnit() {
    return Math.abs(norm() - 1.0) < 1e-10;
  }

  /**
   * Checks if this quaternion is the zero quaternion.
   *
   * @return true if w = x = y = z = 0, false otherwise
   */
  public boolean isZero() {
    return w == 0.0 && x == 0.0 && y == 0.0 && z == 0.0;
  }

  /**
   * Adds another quaternion to this one.
   *
   * @param other the quaternion to add
   * @return a new quaternion representing the sum
   */
  public Quaternion add(Quaternion other) {
    return new Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
  }

  /**
   * Subtracts another quaternion from this one.
   *
   * @param other the quaternion to subtract
   * @return a new quaternion representing the difference
   */
  public Quaternion subtract(Quaternion other) {
    return new Quaternion(w - other.w, x - other.x, y - other.y, z - other.z);
  }

  /**
   * Multiplies this quaternion by another quaternion.
   *
   * <p>Quaternion multiplication is non-commutative and follows the rules: i² = j² = k² = ijk = -1,
   * ij = k, ji = -k, jk = i, kj = -i, ki = j, ik = -j
   *
   * @param other the quaternion to multiply by
   * @return a new quaternion representing the product
   */
  public Quaternion multiply(Quaternion other) {
    return new Quaternion(
        w * other.w - x * other.x - y * other.y - z * other.z,
        w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w);
  }

  /**
   * Multiplies this quaternion by a scalar value.
   *
   * @param scalar the scalar to multiply by
   * @return a new quaternion with all components scaled
   */
  public Quaternion multiply(double scalar) {
    return new Quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
  }

  /**
   * Divides this quaternion by a scalar value.
   *
   * @param scalar the scalar to divide by
   * @return a new quaternion with all components divided
   * @throws IllegalArgumentException if scalar is zero
   */
  public Quaternion divide(double scalar) {
    if (scalar == 0.0) {
      throw new IllegalArgumentException("Division by zero");
    }
    return new Quaternion(w / scalar, x / scalar, y / scalar, z / scalar);
  }

  /**
   * Negates this quaternion (multiplies by -1).
   *
   * @return a new quaternion with all components negated
   */
  public Quaternion negate() {
    return new Quaternion(-w, -x, -y, -z);
  }

  /**
   * Returns the conjugate of this quaternion (negates vector part).
   *
   * @return a new quaternion with vector components negated
   */
  public Quaternion conjugate() {
    return new Quaternion(w, -x, -y, -z);
  }

  /**
   * Returns the norm (magnitude) of this quaternion.
   *
   * @return the square root of w² + x² + y² + z²
   */
  public double norm() {
    return Math.sqrt(w * w + x * x + y * y + z * z);
  }

  /**
   * Returns the squared norm of this quaternion.
   *
   * @return w² + x² + y² + z²
   */
  public double normSquared() {
    return w * w + x * x + y * y + z * z;
  }

  /**
   * Normalizes this quaternion to unit length.
   *
   * @return a new unit quaternion in the same direction
   * @throws IllegalArgumentException if this quaternion is zero
   */
  public Quaternion normalize() {
    double n = norm();
    if (n == 0.0) {
      throw new IllegalArgumentException("Cannot normalize zero quaternion");
    }
    return divide(n);
  }

  /**
   * Returns the multiplicative inverse of this quaternion.
   *
   * @return a new quaternion q⁻¹ such that q × q⁻¹ = 1
   * @throws IllegalArgumentException if this quaternion is zero
   */
  public Quaternion inverse() {
    double n2 = normSquared();
    if (n2 == 0.0) {
      throw new IllegalArgumentException("Cannot invert zero quaternion");
    }
    return conjugate().divide(n2);
  }

  /**
   * Computes the dot product with another quaternion.
   *
   * @param other the quaternion to compute dot product with
   * @return the scalar dot product w₁w₂ + x₁x₂ + y₁y₂ + z₁z₂
   */
  public double dot(Quaternion other) {
    return w * other.w + x * other.x + y * other.y + z * other.z;
  }

  /**
   * Computes the true quaternion cross product.
   *
   * <p>For quaternions q₁ = w₁ + v₁ and q₂ = w₂ + v₂, the quaternion cross product is: q₁ × q₂ =
   * (w₁v₂ - w₂v₁ + v₁ × v₂, w₁w₂ - v₁·v₂)
   *
   * <p>This is the mathematically correct quaternion cross product, not to be confused with the 3D
   * vector cross product of the vector parts.
   *
   * @param other the quaternion to compute cross product with
   * @return a new quaternion representing the quaternion cross product
   * @throws IllegalArgumentException if other is null
   */
  public Quaternion cross(Quaternion other) {
    if (other == null) {
      throw new IllegalArgumentException("Other quaternion cannot be null");
    }

    // Extract components
    double w1 = this.w, w2 = other.w;
    double[] v1 = this.getVector();
    double[] v2 = other.getVector();

    // Compute vector cross product v1 × v2
    double[] vCross = {
      v1[1] * v2[2] - v1[2] * v2[1], // x component
      v1[2] * v2[0] - v1[0] * v2[2], // y component
      v1[0] * v2[1] - v1[1] * v2[0] // z component
    };

    // Compute dot product v1 · v2
    double vDot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

    // Quaternion cross product: (w₁v₂ - w₂v₁ + v₁ × v₂, w₁w₂ - v₁·v₂)
    double newW = w1 * w2 - vDot;
    double[] newV = {
      w1 * v2[0] - w2 * v1[0] + vCross[0],
      w1 * v2[1] - w2 * v1[1] + vCross[1],
      w1 * v2[2] - w2 * v1[2] + vCross[2]
    };

    return new Quaternion(newW, newV);
  }

  /**
   * Computes the 3D vector cross product of the vector parts of two quaternions.
   *
   * <p>This method extracts the vector parts of both quaternions and computes their 3D cross
   * product. The result is a vector quaternion (w = 0) representing the cross product in 3D space.
   *
   * <p><strong>Use Case:</strong> When you need the 3D vector cross product for geometric
   * operations, not the quaternion cross product. For quaternion operations, use <code>cross()
   * </code> instead.
   *
   * @param other the quaternion to compute vector cross product with
   * @return a new vector quaternion representing the 3D vector cross product
   * @throws IllegalArgumentException if other is null
   */
  public Quaternion vectorCross(Quaternion other) {
    if (other == null) {
      throw new IllegalArgumentException("Other quaternion cannot be null");
    }

    // Extract vector parts and compute 3D cross product
    double[] v1 = this.getVector();
    double[] v2 = other.getVector();

    return new Quaternion(
        0.0,
        v1[1] * v2[2] - v1[2] * v2[1], // x component
        v1[2] * v2[0] - v1[0] * v2[2], // y component
        v1[0] * v2[1] - v1[1] * v2[0] // z component
        );
  }

  /**
   * Computes the geodesic distance between this quaternion and another along the great circle on
   * S³.
   *
   * <p>The geodesic distance is the angle between the quaternions, computed using the dot product.
   * For unit quaternions, this represents the shortest angular distance on the 4D unit sphere.
   *
   * <p><strong>Mathematical Definition:</strong> For unit quaternions q₁ and q₂, the geodesic
   * distance is 2 * arccos(|q₁ · q₂|), where · is the dot product. This represents the angle
   * between the quaternions in 4D space.
   *
   * <p><strong>Properties:</strong>
   *
   * <ul>
   *   <li>Distance is always in [0, 2π] radians
   *   <li>Distance = 0 if and only if quaternions are identical
   *   <li>Distance = 2π if and only if quaternions are antipodal (q₁ = -q₂)
   * </ul>
   *
   * @param other the other quaternion
   * @return the geodesic distance in radians
   * @throws IllegalArgumentException if other is null or either quaternion is not normalized
   */
  public double geodesicDistance(Quaternion other) {
    if (other == null) {
      throw new IllegalArgumentException("Other quaternion cannot be null");
    }
    if (!this.isUnit() || !other.isUnit()) {
      throw new IllegalArgumentException("Both quaternions must be normalized (unit quaternions)");
    }

    // For unit quaternions, the geodesic distance is 2 * arccos(|this · other|)
    // where · is the dot product
    double dotProduct = Math.abs(this.dot(other));

    // Clamp to [0, 1] to avoid numerical issues
    dotProduct = Math.max(0.0, Math.min(1.0, dotProduct));

    // Handle the case where quaternions are nearly identical (numerical stability)
    if (dotProduct > 0.9999) {
      return 0.0;
    }

    return 2.0 * Math.acos(dotProduct);
  }

  /**
   * Creates a quaternion representing rotation around an axis.
   *
   * <p>The rotation is specified by an angle (in radians) and a 3D axis vector. The axis vector
   * will be normalized automatically.
   *
   * @param angle the rotation angle in radians
   * @param axis the 3D axis vector [x, y, z]
   * @return a new quaternion representing the rotation
   * @throws IllegalArgumentException if axis does not have exactly 3 components or is zero
   */
  public static Quaternion fromAxisAngle(double angle, double[] axis) {
    if (axis.length != 3) {
      throw new IllegalArgumentException("Axis must have exactly 3 components");
    }

    double norm = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (norm == 0.0) {
      throw new IllegalArgumentException("Axis cannot be zero vector");
    }

    double halfAngle = angle / 2.0;
    double sinHalfAngle = Math.sin(halfAngle);

    return new Quaternion(
        Math.cos(halfAngle),
        axis[0] * sinHalfAngle / norm,
        axis[1] * sinHalfAngle / norm,
        axis[2] * sinHalfAngle / norm);
  }

  /**
   * Creates a quaternion from a rotation vector representation.
   *
   * <p>A rotation vector is a 3D vector where the direction represents the rotation axis and the
   * magnitude represents the rotation angle. This is the inverse operation of toRotationVector().
   *
   * @param rotationVector the rotation vector as a 3D array [x, y, z]
   * @return a quaternion representing the rotation
   * @throws IllegalArgumentException if rotationVector does not have exactly 3 components
   */
  public static Quaternion fromRotationVector(double[] rotationVector) {
    if (rotationVector.length != 3) {
      throw new IllegalArgumentException("Rotation vector must have exactly 3 components");
    }

    // Extract angle and axis from the rotation vector
    double angle =
        Math.sqrt(
            rotationVector[0] * rotationVector[0]
                + rotationVector[1] * rotationVector[1]
                + rotationVector[2] * rotationVector[2]);

    // Handle very small rotations (return identity)
    if (angle < 1e-10) {
      return Quaternion.ONE;
    }

    // Normalize the axis
    double[] axis = {
      rotationVector[0] / angle, rotationVector[1] / angle, rotationVector[2] / angle
    };

    // Create quaternion using axis-angle representation
    return fromAxisAngle(angle, axis);
  }

  /**
   * Performs spherical linear interpolation (SLERP) between two quaternions.
   *
   * <p>SLERP provides smooth interpolation between two rotations, following the shortest arc on the
   * 4D unit sphere. The interpolation parameter t should be in [0, 1].
   *
   * @param q1 the first quaternion
   * @param q2 the second quaternion
   * @param t the interpolation parameter (0.0 = q1, 1.0 = q2)
   * @return a new quaternion interpolated between q1 and q2
   * @throws IllegalArgumentException if t is not in [0, 1]
   */
  public static Quaternion slerp(Quaternion q1, Quaternion q2, double t) {
    if (t < 0.0 || t > 1.0) {
      throw new IllegalArgumentException("Interpolation parameter must be in [0, 1]");
    }

    double dot = q1.dot(q2);
    Quaternion q2Adjusted = q2;

    if (dot < 0.0) {
      q2Adjusted = q2.negate();
      dot = -dot;
    }

    if (dot > 0.9995) {
      return q1.add(q2Adjusted.subtract(q1).multiply(t)).normalize();
    }

    double theta = Math.acos(dot);
    double sinTheta = Math.sin(theta);

    return q1.multiply(Math.sin((1 - t) * theta) / sinTheta)
        .add(q2Adjusted.multiply(Math.sin(t * theta) / sinTheta));
  }

  /**
   * Computes the exponential of this quaternion.
   *
   * <p>For a quaternion q = w + v where v is the vector part: e^q = e^w * (cos(|v|) + v/|v| *
   * sin(|v|))
   *
   * <p><strong>Lie Group Mapping:</strong> For pure vector quaternions (w = 0), this maps from the
   * tangent space (so(3)) to the quaternion group (SO(3)). The result is a unit quaternion
   * representing a rotation around the axis defined by the vector part.
   *
   * <p><strong>Mathematical Properties:</strong>
   *
   * <ul>
   *   <li>If q is a pure vector quaternion (w = 0), then exp(q) is a unit quaternion
   *   <li>exp(q₁ + q₂) ≠ exp(q₁) * exp(q₂) for non-commuting quaternions
   *   <li>exp(0) = 1 (identity quaternion)
   * </ul>
   *
   * @return a new quaternion representing e^q
   */
  public Quaternion exp() {
    double vNorm = Math.sqrt(x * x + y * y + z * z);
    if (vNorm == 0.0) {
      return new Quaternion(Math.exp(w), 0.0, 0.0, 0.0);
    }

    double expW = Math.exp(w);
    double sinV = Math.sin(vNorm);
    double cosV = Math.cos(vNorm);

    return new Quaternion(
        expW * cosV, expW * sinV * x / vNorm, expW * sinV * y / vNorm, expW * sinV * z / vNorm);
  }

  /**
   * Computes the natural logarithm of this quaternion.
   *
   * <p>For a quaternion q = w + v where v is the vector part: ln(q) = ln(|q|) + v/|v| * acos(w/|q|)
   *
   * <p><strong>Lie Algebra Mapping:</strong> For unit quaternions, this maps from the quaternion
   * group (SO(3)) to its tangent space (so(3)). The result is a quaternion where the scalar part
   * represents ln(|q|) and the vector part represents the rotation in tangent space.
   *
   * @return a new quaternion representing ln(q)
   * @throws IllegalArgumentException if this quaternion is zero
   */
  public Quaternion log() {
    double vNorm = Math.sqrt(x * x + y * y + z * z);
    double qNorm = norm();

    if (qNorm == 0.0) {
      throw new IllegalArgumentException("Cannot take log of zero quaternion");
    }

    if (vNorm == 0.0) {
      return new Quaternion(Math.log(qNorm), 0.0, 0.0, 0.0);
    }

    double logNorm = Math.log(qNorm);
    double acosW = Math.acos(w / qNorm);

    return new Quaternion(logNorm, acosW * x / vNorm, acosW * y / vNorm, acosW * z / vNorm);
  }

  /**
   * Raises this quaternion to a scalar power.
   *
   * <p>Uses the identity q^a = e^(a * ln(q)) for general powers. Special cases: q^0 = 1, q^1 = q,
   * q^(-1) = q⁻¹
   *
   * @param exponent the scalar exponent
   * @return a new quaternion representing q^exponent
   */
  public Quaternion pow(double exponent) {
    if (exponent == 0.0) return ONE;
    if (exponent == 1.0) return this;
    if (exponent == -1.0) return inverse();

    return log().multiply(exponent).exp();
  }

  /**
   * Raises this quaternion to a quaternion power.
   *
   * <p>Currently only supports scalar quaternion exponents.
   *
   * @param exponent the quaternion exponent
   * @return a new quaternion representing q^exponent
   * @throws UnsupportedOperationException if exponent is not scalar
   */
  public Quaternion pow(Quaternion exponent) {
    if (exponent.isScalar()) {
      return pow(exponent.w);
    }
    throw new UnsupportedOperationException("Non-scalar exponent not supported");
  }

  /**
   * Checks if this quaternion equals another object.
   *
   * <p>Two quaternions are equal if all their components are equal. Uses Double.compare for proper
   * handling of NaN and signed zeros.
   *
   * @param obj the object to compare with
   * @return true if the objects are equal, false otherwise
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;

    Quaternion that = (Quaternion) obj;
    return Double.compare(that.w, w) == 0
        && Double.compare(that.x, x) == 0
        && Double.compare(that.y, y) == 0
        && Double.compare(that.z, z) == 0;
  }

  /**
   * Returns a hash code for this quaternion.
   *
   * @return a hash code value for this object
   */
  @Override
  public int hashCode() {
    return Objects.hash(w, x, y, z);
  }

  /**
   * Converts this quaternion to its rotation vector representation. This extracts the axis-angle
   * representation and returns the axis scaled by the angle.
   *
   * <p>For unit quaternions, the rotation vector is the vector part scaled by 2*acos(w). This is
   * useful for converting quaternions to 3D rotation vectors for mathematical operations.
   *
   * <p><strong>Important Note:</strong> This method is different from <code>log().getVector()
   * </code>:
   *
   * <ul>
   *   <li><code>toRotationVector()</code>: Returns the rotation vector (axis × angle) in 3D space
   *   <li><code>log().getVector()</code>: Returns the rotation vector in the tangent space (Lie
   *       algebra)
   * </ul>
   *
   * <p>For Lie algebra operations (like in your perceptron), use <code>log().getVector()</code>.
   * For geometric interpretations, use <code>toRotationVector()</code>.
   *
   * @return the rotation vector as a 3D array [x, y, z]
   */
  public double[] toRotationVector() {
    if (!isUnit()) {
      throw new IllegalStateException(
          "Quaternion must be normalized for rotation vector conversion");
    }

    // For unit quaternions, the rotation angle is 2*acos(w)
    double w = this.w;
    double[] vector = getVector();

    // Handle the case where w is very close to 1 (small rotation)
    if (Math.abs(w - 1.0) < NUMERICAL_STABILITY_THRESHOLD) {
      return new double[] {0.0, 0.0, 0.0};
    }

    // Compute the rotation angle
    double angle = 2.0 * Math.acos(Math.max(-1.0, Math.min(1.0, w)));

    // Normalize the vector part and scale by angle
    double vectorNorm =
        Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    if (vectorNorm < NUMERICAL_STABILITY_THRESHOLD) {
      return new double[] {0.0, 0.0, 0.0};
    }

    return new double[] {
      angle * vector[0] / vectorNorm, angle * vector[1] / vectorNorm, angle * vector[2] / vectorNorm
    };
  }

  /**
   * Returns a string representation of this quaternion.
   *
   * <p>The format is "Quaternion(w, x, y, z)" with 6 decimal places precision.
   *
   * @return a string representation of this quaternion
   */
  @Override
  public String toString() {
    return String.format("Quaternion(%.6f, %.6f, %.6f, %.6f)", w, x, y, z);
  }
}
