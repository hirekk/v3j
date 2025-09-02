/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package data;

import java.util.List;

import data.strategy.GenerationStrategy;

/**
 * Orchestrates dataset generation using the Strategy pattern.
 *
 * <p>This class provides an interface for configuring and generating datasets. It uses the Strategy
 * pattern to delegate actual dataset generation to different GenerationStrategy implementations,
 * allowing for flexible generation approaches (e.g., exact XOR, fuzzy XOR).
 *
 * <p>The generator must be configured with both dimensionality and strategy before generating
 * datasets. The interface allows for method chaining during configuration.
 */
public final class DatasetGenerator {
  /** The number of dimensions for generated datasets */
  private int numDimensions;

  /** The strategy to use for dataset generation */
  private GenerationStrategy strategy;

  /**
   * Sets the number of dimensions for generated datasets.
   *
   * @param numDimensions the dimensionality of datasets to generate
   * @return this DatasetGenerator instance for method chaining
   */
  public DatasetGenerator withNumDimensions(int numDimensions) {
    this.numDimensions = numDimensions;
    return this;
  }

  /**
   * Sets the generation strategy to use for dataset creation.
   *
   * @param strategy the GenerationStrategy implementation to use
   * @return this DatasetGenerator instance for method chaining
   */
  public DatasetGenerator withStrategy(GenerationStrategy strategy) {
    this.strategy = strategy;
    return this;
  }

  /**
   * Generates a dataset using the configured strategy and dimensionality.
   *
   * <p>This method delegates the actual dataset generation to the configured GenerationStrategy
   * implementation.
   *
   * @return a list of DataPoint objects representing the generated dataset
   * @throws IllegalStateException if no strategy has been configured
   */
  public List<DataPoint> generateDataset() {
    if (strategy == null) {
      throw new IllegalStateException("Must set strategy to generate a dataset");
    }
    return strategy.generateDataset(numDimensions);
  }

  /**
   * Returns the currently configured number of dimensions.
   *
   * @return the number of dimensions for generated datasets
   */
  public int getNumDimensions() {
    return numDimensions;
  }
}
