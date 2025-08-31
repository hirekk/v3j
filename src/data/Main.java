package data;

import data.strategy.ExactXorStrategy;
import data.strategy.FuzzyXorStrategy;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Command-line interface for generating XOR-style datasets.
 *
 * <p>This class provides a CLI with two subcommands for generating different types of XOR datasets:
 * exact hypercube vertices and fuzzy data clouds. The interface uses Picocli for argument parsing
 * and provides help options for each command.
 *
 * <p>The main class acts as a command dispatcher, while the actual dataset generation logic is
 * implemented in the subcommand classes.
 *
 * @see ExactCommand
 * @see FuzzyCommand
 */
@Command(
    name = "xor-generator",
    mixinStandardHelpOptions = true,
    subcommands = {ExactCommand.class, FuzzyCommand.class})
public class Main {

  /**
   * Main entry point for the XOR dataset generator CLI.
   *
   * <p>Parses command-line arguments and executes the appropriate subcommand. Exits with the return
   * code from the executed command.
   *
   * @param args command-line arguments to parse
   */
  public static void main(String[] args) {
    int exitCode = new CommandLine(new Main()).execute(args);
    System.exit(exitCode);
  }
}

/**
 * Subcommand for generating exact XOR datasets.
 *
 * <p>Generates datasets containing exactly the vertices of an N-dimensional hypercube. Each vertex
 * is labeled based on the parity of 1s in its coordinates (odd = 1, even = 0). The dataset size is
 * always 2^N where N is the number of dimensions.
 */
@Command(
    name = "exact",
    mixinStandardHelpOptions = true,
    description = "Generate exact XOR dataset")
class ExactCommand implements Runnable {

  /** Number of dimensions for the hypercube */
  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  /** Output file path for the generated CSV dataset */
  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/exact.csv";

  /**
   * Executes the exact XOR dataset generation.
   *
   * <p>Creates a DatasetGenerator with ExactXorStrategy, generates the dataset, and exports it to
   * the specified CSV file.
   *
   * @throws RuntimeException if dataset generation or export fails
   */
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

/**
 * Subcommand for generating fuzzy XOR datasets.
 *
 * <p>Generates datasets with data clouds around each hypercube vertex. Each vertex has a
 * configurable number of points distributed with Gaussian noise around it. The dataset size is 2^N
 * × cardinality where N is the number of dimensions.
 */
@Command(
    name = "fuzzy",
    mixinStandardHelpOptions = true,
    description = "Generate fuzzy XOR data clouds")
class FuzzyCommand implements Runnable {

  /** Number of dimensions for the hypercube */
  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  /** Number of points to generate around each vertex */
  @Option(
      names = {"--blob-cardinality", "-c"},
      required = true,
      description = "Number of points per vertex cloud")
  private int blobCardinality;

  /** Standard deviation for the Gaussian distribution around vertices */
  @Option(
      names = {"--blob-variance", "-v"},
      required = true,
      description = "Blob variance")
  private double blobVariance;

  /** Random seed for reproducible generation (optional) */
  @Option(
      names = {"--seed", "-s"},
      required = false,
      description = "Random seed")
  private Long seed = 0L;

  /** Output file path for the generated CSV dataset */
  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/fuzzy.csv";

  /**
   * Executes the fuzzy XOR dataset generation.
   *
   * <p>Creates a DatasetGenerator with FuzzyXorStrategy, generates the dataset with configurable
   * cardinality and variance, and exports it to the specified CSV file.
   *
   * @throws RuntimeException if dataset generation or export fails
   */
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
