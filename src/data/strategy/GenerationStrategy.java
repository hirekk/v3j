/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package data.strategy;

import java.util.List;

import data.DataPoint;
import data.DatasetGenerator;

/**
 * Strategy interface for generating datasets with different algorithms and characteristics.
 *
 * <p>This interface defines the contract for dataset generation strategies, allowing different
 * implementations to be used interchangeably with the {@link DatasetGenerator}. Each strategy
 * encapsulates a specific algorithm for creating data points with particular properties,
 * distributions, or mathematical relationships.
 *
 * <p>Strategies can implement various approaches such as:
 *
 * <ul>
 *   <li><strong>Exact algorithms</strong> - Deterministic generation of specific mathematical
 *       structures (e.g., hypercube vertices)
 *   <li><strong>Probabilistic algorithms</strong> - Generation with controlled randomness and
 *       statistical properties (e.g., Gaussian noise around centers)
 *   <li><strong>Pattern-based algorithms</strong> - Generation following specific rules or
 *       mathematical relationships
 * </ul>
 *
 * <p>Implementations should be immutable and thread-safe to allow safe concurrent usage across
 * multiple threads.
 *
 * @see DatasetGenerator
 * @see DataPoint
 * @see ExactXorStrategy
 * @see FuzzyXorStrategy
 */
public interface GenerationStrategy {

  /**
   * Generates a dataset according to the strategy's specific algorithm.
   *
   * <p>This method creates a collection of {@link DataPoint} objects that conform to the strategy's
   * generation rules. The exact number of points, their coordinate values, and their labels depend
   * on the specific implementation.
   *
   * <p>Implementations should ensure that:
   *
   * <ul>
   *   <li>All generated points have the specified number of dimensions
   *   <li>Labels are consistent with the strategy's labeling logic
   *   <li>The dataset size is appropriate for the given dimensions
   *   <li>Any random elements are properly seeded for reproducibility
   * </ul>
   *
   * @param numDimensions the number of dimensions for each data point
   * @return a list of DataPoint objects generated according to the strategy
   * @throws IllegalArgumentException if numDimensions is invalid for this strategy
   */
  List<DataPoint> generateDataset(int numDimensions);
}
