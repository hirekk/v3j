package data;

import data.strategy.GenerationStrategy;
import java.util.List;

public final class DatasetGenerator {
  private int numDimensions;
  private GenerationStrategy strategy;

  public DatasetGenerator withNumDimensions(int numDimensions) {
    this.numDimensions = numDimensions;
    return this;
  }

  public DatasetGenerator withStrategy(GenerationStrategy strategy) {
    this.strategy = strategy;
    return this;
  }

  public List<DataPoint> generateDataset() {
    if (strategy == null) {
      throw new IllegalStateException("Must set strategy to generate a dataset");
    }
    return strategy.generateDataset(numDimensions);
  }

  public int getNumDimensions() {
    return numDimensions;
  }
}
