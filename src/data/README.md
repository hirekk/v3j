# Data Package

Generates N-dimensional XOR-style datasets for machine learning and testing purposes.

## Components

- **DataPoint**: Immutable record representing a single data point with coordinates and binary label
- **Dataset**: Container for data points with CSV import/export, shuffling, and subsetting capabilities
- **DatasetGenerator**: Orchestrates dataset generation using the Strategy pattern
- **Main**: CLI interface with subcommands for exact and fuzzy dataset generation

## Generation Strategies

- **ExactXorStrategy**: Generates exact hypercube vertices (2^N points)
- **FuzzyXorStrategy**: Generates data clouds around vertices with configurable cardinality and variance

## Usage

### CLI Commands

Generate exact XOR dataset:
```bash
./gradlew run --args="exact -d 3 -o data/exact_3d.csv"
```

Generate fuzzy XOR dataset:
```bash
./gradlew run --args="fuzzy -d 2 -c 10 -v 0.1 -s 42 -o data/fuzzy_2d.csv"
```

### CLI Options Reference

**Exact Command:**
- `-d, --num-dimensions` (required): Number of dimensions
- `-o, --output-path`: Output CSV file path (default: `data/xor/exact.csv`)

**Fuzzy Command:**
- `-d, --num-dimensions` (required): Number of dimensions
- `-c, --blob-cardinality` (required): Number of points per vertex cloud
- `-v, --blob-variance` (required): Blob variance
- `-s, --seed`: Random seed (default: 0)
- `-o, --output-path`: Output CSV file path (default: `data/xor/fuzzy.csv`)

### Programmatic Usage

```java
// Create exact dataset
DatasetGenerator generator = new DatasetGenerator()
    .withNumDimensions(3)
    .withStrategy(new ExactXorStrategy());
Dataset dataset = Dataset.fromGenerator(generator);

// Export to CSV
dataset.exportToCSV(Path.of("data/xor/data.csv"));

// Import from CSV
Dataset imported = Dataset.fromCsv(Path.of("data/xor/data.csv"));
```

## Dataset Types

**Exact XOR**: Generates vertices of N-dimensional hypercube with binary labels based on coordinate parity.

**Fuzzy XOR**: Generates configurable number of points around each vertex with Gaussian noise distribution.

## File Format

CSV files contain N coordinate columns followed by a binary label column (0 or 1).

## Example Usage

**Create a 3D exact XOR dataset:**
```bash
./gradlew run --args="exact -d 3 -o data/xor/exact3d.csv"
```

**Create a 2D fuzzy XOR dataset:**
```bash
./gradlew run --args="fuzzy -d 2 -c 20 -v 0.1 -o data/xor/fuzzy2d.csv"
```

**Create a 3D fuzzy XOR dataset with custom seed:**
```bash
./gradlew run --args="fuzzy -d 3 -c 2 -v 0.01 -s 42 -o data/xor/fuzzy3d.csv"
```
