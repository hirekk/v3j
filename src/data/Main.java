package data;

import data.strategy.ExactXorStrategy;
import data.strategy.FuzzyXorStrategy;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(
    name = "xor-generator",
    mixinStandardHelpOptions = true,
    subcommands = {ExactCommand.class, FuzzyCommand.class})
public class Main {
  public static void main(String[] args) {
    int exitCode = new CommandLine(new Main()).execute(args);
    System.exit(exitCode);
  }
}

@Command(
    name = "exact",
    mixinStandardHelpOptions = true,
    description = "Generate exact XOR dataset")
class ExactCommand implements Runnable {

  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/exact.csv";

  @Override
  public void run() {
    DatasetGenerator generator =
        new DatasetGenerator()
            .withNumDimensions(numDimensions)
            .withStrategy(new ExactXorStrategy());

    Dataset dataset = Dataset.fromGenerator(generator);
    try {
      dataset.exportToCSV(Path.of(outputPath));
    } catch (IOException e) {
      System.err.printf("Error saving dataset: %s%n", e.getMessage());
      System.exit(1);
    }

    System.out.printf(
        "Generated exact XOR dataset with %d dimensions, saved to %s%n", numDimensions, outputPath);
  }
}

@Command(
    name = "fuzzy",
    mixinStandardHelpOptions = true,
    description = "Generate fuzzy XOR data clouds")
class FuzzyCommand implements Runnable {

  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  @Option(
      names = {"--blob-cardinality", "-c"},
      required = true,
      description = "Number of points per vertex cloud")
  private int blobCardinality;

  @Option(
      names = {"--blob-variance", "-v"},
      required = true,
      description = "Blob variance")
  private double blobVariance;

  @Option(
      names = {"--seed", "-s"},
      required = false,
      description = "Random seed")
  private Long seed = 0L;

  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/fuzzy.csv";

  @Override
  public void run() {
    double[] blobVarianceArray = new double[numDimensions];
    Arrays.fill(blobVarianceArray, blobVariance);
    DatasetGenerator generator =
        new DatasetGenerator()
            .withNumDimensions(numDimensions)
            .withStrategy(new FuzzyXorStrategy(blobCardinality, blobVarianceArray, seed));

    Dataset dataset = Dataset.fromGenerator(generator);
    try {
      dataset.exportToCSV(Path.of(outputPath));
    } catch (IOException e) {
      System.err.printf("Error saving dataset: %s%n", e.getMessage());
      System.exit(1);
    }

    System.out.printf(
        "Generated fuzzy XOR dataset with %d dimensions, saved to %s%n", numDimensions, outputPath);
  }
}
