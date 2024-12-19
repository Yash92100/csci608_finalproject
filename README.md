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
