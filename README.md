# Electric Demand Forecast

A deep learning project for predicting regional and metropolitan electricity consumption in France for 2022, using historical weather data and electricity consumption from 2017-2021.

## Project Overview

This project uses neural networks to forecast electricity demand across 13 French regions and 12 metropolitan areas. The models are trained on:
- Historical electricity consumption data (2017-2021)
- Weather measurements from multiple meteorological stations
- Temporal features (holidays, time of day, day of week, seasonality)
- Population-weighted regional aggregations

## Key Features

- **MLP Model**: A custom Multi-Layer Perceptron architecture optimized for electricity demand forecasting
- **DRAGON AutoML**: Neural Architecture Search (NAS) using the DRAGON library for automated model optimization
- **Weather Processing**: Sophisticated weather data aggregation and interpolation at 30-minute intervals
- **Regional & Metropolitan Coverage**: Predictions for 25 geographic entities across France

## Project Structure

```
ElectricDemandForecast/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── prepare_data.py               # Data initialization script
├── traitement_meteo.py           # Weather data processing
├── model.py                      # Neural network architectures
├── train.py                      # Model training pipeline
├── predict.py                    # Prediction generation
├── dragon_model.py               # DRAGON AutoML implementation
├── .gitignore                    # Git ignore rules
│
├── Data/                         # Dataset files (created after setup)
│   ├── jours_feries_metropole.csv
│   ├── population_region.csv
│   ├── meteo.parquet
│   ├── X_train_final.csv
│   ├── X_2022_final.csv
│   └── y_train.csv
│
├── Model/                        # Trained model artifacts
│   ├── pred_model.pth
│   ├── model_params.pt
│   ├── scaler_x.pt
│   ├── scaler_y.pt
│   ├── total_columns.pt
│   └── X_2022_prepared.csv
│
├── save/                         # DRAGON model checkpoints
│   └── test_mutant/
│       └── best_model/
│           ├── x.pkl
│           └── best_model.pth
│
└── Solutions/                    # Prediction outputs
    └── pred.csv
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git LFS (for large data files)
- CUDA-compatible GPU (optional, but recommended for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EpsilonFO/ElectricDemandForecast.git
   cd ElectricDemandForecast
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the project**
   ```bash
   python prepare_data.py
   ```
   
   This script will:
   - Pull large files using Git LFS
   - Create necessary directories (`Model/`, `Solutions/`, `save/`)
   - Move data files to their appropriate locations

### Usage

#### Standard MLP Approach (Recommended)

**Step 1: Train the model**
```bash
python train.py
```

This will:
- Load and preprocess training data
- Train the MLP model for 50 epochs
- Save the trained model to `Model/pred_model.pth`
- Save scalers and metadata for prediction
- Display training progress and final loss

Expected training time: 5-15 minutes (depending on hardware)

**Step 2: Generate predictions**
```bash
python predict.py
```

This will:
- Load the trained model
- Generate predictions for 2022
- Save results to `Solutions/pred.csv`

#### DRAGON AutoML Approach (Advanced)

For automated neural architecture search:

```bash
python dragon_model.py
```

**Warning**: This approach is computationally intensive and can take several hours to complete. Pre-trained DRAGON models are included in `save/test_mutant/best_model/` for convenience.

The DRAGON approach:
- Performs 200 iterations of architecture search
- Evaluates multiple neural network configurations
- Automatically selects the best performing architecture
- Saves the optimal model for inference

## Model Architecture

### MLP Model (`model.py`)

The custom MLP uses:
- Input layer: Scaled weather and temporal features
- Hidden layers with ELU, GELU, and Tanh activations
- Output layer: 25 neurons (13 regions + 12 metropolitan areas)
- Loss function: Region-wise RMSE with NaN handling

### DRAGON MetaArchi (`dragon_model.py`)

The DRAGON model implements:
- Dynamic neural architecture search
- DAG-based architecture encoding
- Mutant-UCB search algorithm
- Automatic hyperparameter optimization

## Data Processing

### Weather Data (`traitement_meteo.py`)

The weather processing pipeline:
1. Aggregates meteorological stations by region (averaging)
2. Pivots data to create region-specific columns
3. Resamples to 30-minute intervals with linear interpolation
4. Adds holiday indicators
5. Creates population-weighted national aggregates
6. Removes daylight saving time anomalies
7. Splits into training (2017-2021) and test (2022) sets

Processed features include:
- Temperature (`t`), humidity (`u`), wind speed (`ff`)
- Atmospheric pressure (`pres`), precipitation (`rr1`, `rr3`, `rr6`)
- Cloud coverage (`nnuage1`, `hnuage1`)
- And 10 additional meteorological variables

### Feature Engineering

Temporal features:
- One-hot encoded: month, hour, weekday
- Binary: holiday indicator
- Continuous: minute, day

## Results

The model optimizes for regional RMSE (Root Mean Square Error), with special handling for:
- Missing values in consumption data
- Different consumption patterns across regions
- Seasonal and weekly variations
- Holiday effects

## File Descriptions

| File | Purpose |
|------|---------|
| `prepare_data.py` | Initializes project structure and moves data files |
| `traitement_meteo.py` | Processes raw weather data into ML-ready format |
| `model.py` | Defines MLP and MetaArchi neural network classes |
| `train.py` | Training pipeline with data loading and model optimization |
| `predict.py` | Loads trained model and generates 2022 predictions |
| `dragon_model.py` | Implements Neural Architecture Search with DRAGON |

## Troubleshooting

**Issue**: `FileNotFoundError` for data files
- **Solution**: Ensure you've run `prepare_data.py` and have Git LFS installed

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in `train.py` or `dragon_model.py`

**Issue**: Slow training on CPU
- **Solution**: Consider using a GPU or reducing `num_epochs`

**Issue**: DRAGON model takes too long
- **Solution**: Use the pre-trained DRAGON model in `save/test_mutant/best_model/`

## Dependencies

Core libraries:
- PyTorch: Deep learning framework
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Data preprocessing and scaling
- dragon-autodl: Neural architecture search
- graphviz: Model visualization

See `requirements.txt` for complete list.

## Contributors

- [@LylianChallier](https://github.com/LylianChallier)
- [@EpsilonFO](https://github.com/EpsilonFO)
