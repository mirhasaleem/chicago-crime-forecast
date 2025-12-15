## Chicago Crime Rate Forecasting with Prophet

This project predicts crime rates in the City of Chicago using historical crime data and Facebook's `prophet` time-series forecasting library. It includes both an exploratory Jupyter notebook and a fully scripted pipeline in `chicago_crime_forecast.py`.

### Project Structure

- **`Project 3 - Predict Crime Rate in Chicago.ipynb`**: Step‑by‑step notebook walking through EDA, resampling, and Prophet modeling.
- **`chicago_crime_forecast.py`**: A reusable pipeline built around the `ChicagoCrimeForecaster` class, handling:
  - Loading and merging raw CSV files (2005–2017)
  - Cleaning and preprocessing
  - Exploratory visualizations (crime types, locations, temporal trends)
  - Preparing monthly aggregates for Prophet
  - Training a Prophet model and generating forecasts
  - Visualizing forecasts and components
  - Evaluating model performance (MAE, RMSE, MAPE)
  - Saving forecast results and serialized model
- **`requirements.txt`**: Python package requirements for running the project.

### Data

The project uses the *Crimes in Chicago* dataset from the Chicago Police Department’s CLEAR system.

- **Source**: Kaggle – `Crimes in Chicago` (currie32)
- **URL**: see Kaggle page: `https://www.kaggle.com/currie32/crimes-in-chicago`
- **Files expected (placed in a `data/` directory for the script)**:
  - `Chicago_Crimes_2005_to_2007.csv`
  - `Chicago_Crimes_2008_to_2011.csv`
  - `Chicago_Crimes_2012_to_2017.csv`

> Note: The notebook reads the CSVs from the current directory, while the Python script expects them in a `data/` subfolder. Adjust paths as needed.

### Environment Setup

1. **Create and activate a virtual environment (recommended)**:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows
```

2. **Install dependencies**:

From the project root:

```bash
pip install -r requirements.txt
```

If you encounter issues with Prophet, you may need to install it explicitly:

```bash
pip install prophet==1.1.5
```

or (with Conda):

```bash
conda install -c conda-forge prophet
```

### Running the End‑to‑End Script

1. Ensure the three CSV files are downloaded from Kaggle and placed inside a `data/` directory in the project root:

```text
Project 3/
  chicago_crime_forecast.py
  Project 3 - Predict Crime Rate in Chicago.ipynb
  requirements.txt
  data/
    Chicago_Crimes_2005_to_2007.csv
    Chicago_Crimes_2008_to_2011.csv
    Chicago_Crimes_2012_to_2017.csv
```

2. Run the pipeline:

```bash
python chicago_crime_forecast.py
```

The script will:

- Load and explore the data
- Clean and aggregate crimes by time
- Train a Prophet model
- Forecast crime counts for the next 365 days
- Visualize forecast and components
- Evaluate performance on the historical period
- Save outputs under a `results/` directory:
  - `results/chicago_crime_forecast.csv` – full forecast table
  - `results/prophet_model.json` – serialized Prophet model

### Using the `ChicagoCrimeForecaster` Class Interactively

If you want to use the forecaster in your own scripts or a Python shell:

```python
from chicago_crime_forecast import ChicagoCrimeForecaster

forecaster = ChicagoCrimeForecaster(data_dir="data")
forecaster.load_data()
forecaster.clean_data()
forecaster.analyze_crime_patterns()
forecaster.prepare_prophet_data()
forecaster.build_prophet_model()
forecaster.make_forecast(periods=365)
forecaster.visualize_forecast()
metrics = forecaster.evaluate_model()
forecaster.save_results(output_dir="results")
```

### Notebook Workflow

To explore the notebook version:

```bash
jupyter notebook "Project 3 - Predict Crime Rate in Chicago.ipynb"
```

The notebook walks through:

- Problem context and dataset description
- Loading and concatenating yearly CSVs
- Handling missing values and dropping unused columns
- Setting a datetime index and resampling
- EDA plots for crime types, locations, and temporal trends
- Building and running a Prophet forecast on aggregated crime counts

### Troubleshooting

- **FileNotFoundError for CSVs**: Verify they are downloaded from Kaggle and placed in the correct directory (`data/` for the script, or adjust paths).
- **Prophet installation issues**: Try installing via Conda (`conda install -c conda-forge prophet`) or pinning the version as in the command above.
- **Long runtimes or memory usage**: The full dataset (2005–2017) is large. Consider subsetting by date range, crime type, or location if resources are limited.


