# Ibis vs. Pandas on NYC Taxi Dataset & JAX for Classification

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
  - [Ibis](#ibis)
  - [JAX](#jax)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

This project demonstrates how to handle and analyze a large-scale dataset using **Ibis** and **Pandas**. It also explores machine learning model implementation using **JAX** to show its performance advantages and flexibility. The goal is to compare Ibis and Pandas for data handling and implement logistic regression and neural networks using JAX.

### Objectives
1. Compare Ibis and Pandas for data handling and processing.
2. Perform classification tasks on the NYC Taxi dataset using logistic regression and an MLP in JAX.
3. Provide clear visual and numerical comparisons for performance metrics.

---

## Dataset

### NYC Taxi Trip Data (January 2021)

This dataset contains information about NYC yellow taxi trips for January 2021, including timestamps, distances, fares, tips, and passenger counts.

- **Download Link:** [NYC Yellow Taxi Trip Data (January 2021)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet)
- **File Format:** `.parquet`
- **Size:** ~1.5 GB

### Dataset Features
- `tpep_pickup_datetime`: Pickup timestamp.
- `tpep_dropoff_datetime`: Dropoff timestamp.
- `trip_distance`: Distance of the trip in miles.
- `fare_amount`: Fare amount in USD.
- `tip_amount`: Tip amount in USD.
- `passenger_count`: Number of passengers.

---

## Technologies Used

### Ibis

**Ibis** is a library for writing database queries in Python. It supports multiple backends, including DuckDB, BigQuery, and PostgreSQL. Ibis is ideal for processing large datasets efficiently by pushing computations to the backend database.

#### Key Features of Ibis
- **Backend Agnostic:** Supports multiple databases (DuckDB, BigQuery, etc.).
- **Lazy Execution:** Queries are executed only when `.execute()` is called.
- **Pandas-like Syntax:** Easy to learn and use.
- **Scalability:** Handles large datasets efficiently using database engines.

### JAX

**JAX** is a high-performance numerical computing library for Python. It offers:
- **Automatic Differentiation:** Simplifies gradient-based optimizations.
- **Just-in-Time Compilation (JIT):** Speeds up computations.
- **Vectorization:** Efficiently applies operations on large arrays.
- **GPU/TPU Support:** Leverages accelerators for high performance.

JAX is widely used for machine learning, scientific computing, and custom optimization tasks.

---

## Installation

### Prerequisites
- **Python 3.7+**
- **pip** for managing packages
- **Jupyter Notebook** or **JupyterLab**

### Step-by-Step Instructions

1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/ibis-pandas-jax-project.git
   cd ibis-pandas-jax-project
   ```

2. Set Up a Virtual Environment (Optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install --upgrade pip
   pip install ibis-framework duckdb-engine jax jaxlib numpy pandas pyarrow matplotlib scikit-learn optax
   ```

4. Download the Dataset:
   - [NYC Yellow Taxi Data (January 2021)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet)
   - Save the file as `yellow_tripdata_2021-01.parquet` in the root project directory.

---

## Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the Notebook:
   - Locate `ibis_pandas_jax_comparison.ipynb`.
   - Open the notebook.

3. Execute the Cells:
   - Run the cells sequentially using `Shift + Enter`.

---

## Project Structure

```
ibis-pandas-jax-project/
├── yellow_tripdata_2021-01.parquet  # Dataset
├── ibis_pandas_jax_comparison.ipynb  # Jupyter Notebook
├── README.md  # Project details
└── requirements.txt  # Optional dependencies file
```

---

## Results

### Key Metrics
- **Scikit-learn Logistic Regression:** Baseline model using scikit-learn's implementation.
- **JAX Logistic Regression:** Custom implementation leveraging JAX's automatic differentiation.
- **JAX MLP:** A neural network model with one hidden layer.

#### Performance Comparison
| **Model**                   | **Accuracy** | **F1-Score** |
|-----------------------------|--------------|--------------|
| Scikit-learn Logistic       | 0.82         | 0.75         |
| JAX Logistic Regression     | 0.80         | 0.73         |
| JAX MLP                     | 0.85         | 0.78         |

### Visualizations
- **Confusion Matrices** for each model.
- **Bar Charts** comparing accuracy and F1-scores.

---

## Conclusion

This project highlights:
1. **Efficiency of Ibis:**  
   - Scalable and optimized for large datasets.  
   - Outperforms Pandas for large-scale queries.

2. **Power of JAX:**  
   - Enables custom machine learning models with better performance on GPUs/TPUs.  
   - Flexible, with support for advanced optimizations.

---

## References

- [Ibis Documentation](https://ibis-project.org/docs/)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [NYC Taxi Dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [DuckDB Documentation](https://duckdb.org/docs/)
