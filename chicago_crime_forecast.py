"""
Chicago Crime Forecasting using Facebook Prophet
Time series forecasting of crime rates in Chicago from 2005-2017
Dataset: Chicago Police Department CLEAR system records
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Facebook Prophet for time series forecasting
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ChicagoCrimeForecaster:
    """
    Time Series Forecasting of Chicago Crime Data using Facebook Prophet
    """
    
    def __init__(self, data_dir='data'):
        """Initialize the forecaster with data directory"""
        self.data_dir = data_dir
        self.chicago_df = None
        self.prophet_df = None
        self.model = None
        self.forecast = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self):
        """Load and combine Chicago crime datasets"""
        print("Loading Chicago crime datasets...")
        
        try:
            # Load individual datasets
            df_2005_2007 = pd.read_csv(
                f'{self.data_dir}/Chicago_Crimes_2005_to_2007.csv',
                low_memory=False
            )
            df_2008_2011 = pd.read_csv(
                f'{self.data_dir}/Chicago_Crimes_2008_to_2011.csv',
                low_memory=False
            )
            df_2012_2017 = pd.read_csv(
                f'{self.data_dir}/Chicago_Crimes_2012_to_2017.csv',
                low_memory=False
            )
            
            # Combine all datasets
            self.chicago_df = pd.concat(
                [df_2005_2007, df_2008_2011, df_2012_2017],
                ignore_index=True
            )
            
            print(f"✓ Combined dataset shape: {self.chicago_df.shape}")
            print(f"✓ Date range: {self.chicago_df['Date'].min()} to {self.chicago_df['Date'].max()}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the following files are in the 'data' directory:")
            print("1. Chicago_Crimes_2005_to_2007.csv")
            print("2. Chicago_Crimes_2008_to_2011.csv")
            print("3. Chicago_Crimes_2012_to_2017.csv")
            print("\nDownload from: https://www.kaggle.com/currie32/crimes-in-chicago")
            return None
        
        return self.chicago_df
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        if self.chicago_df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Display basic info
        print("\n1. Dataset Overview:")
        print(self.chicago_df.info())
        
        print("\n2. First 5 rows:")
        print(self.chicago_df.head())
        
        print("\n3. Last 5 rows:")
        print(self.chicago_df.tail())
        
        # Check for missing values
        print("\n4. Missing Values Heatmap:")
        self.plot_missing_values()
        
        # Display summary statistics
        print("\n5. Summary Statistics:")
        print(self.chicago_df.describe(include='all'))
    
    def plot_missing_values(self):
        """Plot heatmap of missing values"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.chicago_df.isnull(), 
                   cbar=False, 
                   cmap='viridis',
                   yticklabels=False)
        plt.title('Missing Values in Chicago Crime Dataset', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.chicago_df is None:
            print("Error: Data not loaded.")
            return
        
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        # Drop unnecessary columns
        columns_to_drop = [
            'Unnamed: 0', 'Case Number', 'IUCR', 
            'X Coordinate', 'Y Coordinate', 'Updated On',
            'Year', 'FBI Code', 'Beat', 'Ward', 
            'Community Area', 'Location', 'District',
            'Latitude', 'Longitude'
        ]
        
        # Keep only columns that exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop 
                          if col in self.chicago_df.columns]
        
        print(f"Dropping columns: {columns_to_drop}")
        self.chicago_df.drop(columns=columns_to_drop, inplace=True)
        
        # Convert Date to datetime
        print("Converting Date column to datetime...")
        self.chicago_df['Date'] = pd.to_datetime(
            self.chicago_df['Date'], 
            format='%m/%d/%Y %I:%M:%S %p',
            errors='coerce'
        )
        
        # Set Date as index
        self.chicago_df.set_index('Date', inplace=True)
        
        print(f"✓ Cleaned dataset shape: {self.chicago_df.shape}")
        print(f"✓ Columns remaining: {list(self.chicago_df.columns)}")
        
        return self.chicago_df
    
    def analyze_crime_patterns(self):
        """Analyze crime patterns and trends"""
        if self.chicago_df is None:
            print("Error: Data not cleaned.")
            return
        
        print("\n" + "="*60)
        print("CRIME PATTERN ANALYSIS")
        print("="*60)
        
        # 1. Top crime types
        print("\n1. Top 15 Crime Types:")
        top_crimes = self.chicago_df['Primary Type'].value_counts().head(15)
        print(top_crimes)
        
        # Plot top crime types
        self.plot_top_crime_types(top_crimes)
        
        # 2. Top locations
        print("\n2. Top 15 Crime Locations:")
        top_locations = self.chicago_df['Location Description'].value_counts().head(15)
        print(top_locations)
        
        # Plot top locations
        self.plot_top_locations(top_locations)
        
        # 3. Time-based analysis
        print("\n3. Crime Trends Over Time:")
        self.analyze_temporal_trends()
    
    def plot_top_crime_types(self, top_crimes):
        """Plot top crime types"""
        plt.figure(figsize=(14, 8))
        top_crimes.plot(kind='barh')
        plt.title('Top 15 Crime Types in Chicago (2005-2017)', fontsize=16)
        plt.xlabel('Number of Incidents', fontsize=12)
        plt.ylabel('Crime Type', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_top_locations(self, top_locations):
        """Plot top crime locations"""
        plt.figure(figsize=(14, 8))
        top_locations.plot(kind='barh')
        plt.title('Top 15 Crime Locations in Chicago (2005-2017)', fontsize=16)
        plt.xlabel('Number of Incidents', fontsize=12)
        plt.ylabel('Location', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def analyze_temporal_trends(self):
        """Analyze crime trends over different time periods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Yearly trend
        yearly_counts = self.chicago_df.resample('Y').size()
        axes[0, 0].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Crimes Per Year', fontsize=14)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Crimes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monthly trend
        monthly_counts = self.chicago_df.resample('M').size()
        axes[0, 1].plot(monthly_counts.index, monthly_counts.values, linewidth=1)
        axes[0, 1].set_title('Crimes Per Month', fontsize=14)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Number of Crimes')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Quarterly trend
        quarterly_counts = self.chicago_df.resample('Q').size()
        axes[1, 0].plot(quarterly_counts.index, quarterly_counts.values, marker='s', linewidth=2)
        axes[1, 0].set_title('Crimes Per Quarter', fontsize=14)
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Number of Crimes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Daily trend (sample)
        daily_counts_sample = self.chicago_df.resample('D').size().head(100)
        axes[1, 1].plot(daily_counts_sample.index, daily_counts_sample.values, linewidth=1)
        axes[1, 1].set_title('Crimes Per Day (Sample: First 100 days)', fontsize=14)
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Number of Crimes')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Chicago Crime Trends (2005-2017)', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def prepare_prophet_data(self):
        """Prepare data for Facebook Prophet"""
        if self.chicago_df is None:
            print("Error: Data not cleaned.")
            return
        
        print("\n" + "="*60)
        print("PREPARING DATA FOR PROPHET")
        print("="*60)
        
        # Resample to monthly frequency
        monthly_counts = self.chicago_df.resample('M').size().reset_index()
        monthly_counts.columns = ['Date', 'Crime Count']
        
        # Create Prophet-compatible dataframe
        self.prophet_df = monthly_counts.rename(columns={
            'Date': 'ds',
            'Crime Count': 'y'
        })
        
        print(f"✓ Prophet dataset shape: {self.prophet_df.shape}")
        print(f"✓ Date range: {self.prophet_df['ds'].min()} to {self.prophet_df['ds'].max()}")
        print("\nFirst 5 rows of Prophet data:")
        print(self.prophet_df.head())
        
        return self.prophet_df
    
    def build_prophet_model(self, seasonality_mode='additive', 
                           yearly_seasonality=True, 
                           weekly_seasonality=True, 
                           daily_seasonality=False):
        """Build and train Facebook Prophet model"""
        if self.prophet_df is None:
            print("Error: Prophet data not prepared.")
            return
        
        print("\n" + "="*60)
        print("BUILDING PROPHET MODEL")
        print("="*60)
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add custom seasonality if needed
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Fit the model
        print("Training Prophet model...")
        self.model.fit(self.prophet_df)
        
        print("✓ Model trained successfully")
        print(f"✓ Seasonality mode: {seasonality_mode}")
        print(f"✓ Yearly seasonality: {yearly_seasonality}")
        print(f"✓ Weekly seasonality: {weekly_seasonality}")
        
        return self.model
    
    def make_forecast(self, periods=365):
        """Make future predictions"""
        if self.model is None:
            print("Error: Model not built.")
            return
        
        print("\n" + "="*60)
        print("MAKING FORECASTS")
        print("="*60)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq='D',
            include_history=True
        )
        
        # Make predictions
        print(f"Making predictions for {periods} days into the future...")
        self.forecast = self.model.predict(future)
        
        print(f"✓ Forecast shape: {self.forecast.shape}")
        print(f"✓ Forecast range: {self.forecast['ds'].min()} to {self.forecast['ds'].max()}")
        
        # Display forecast summary
        print("\nForecast Summary:")
        print(self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        return self.forecast
    
    def visualize_forecast(self):
        """Visualize forecast results"""
        if self.forecast is None or self.model is None:
            print("Error: No forecast available.")
            return
        
        print("\n" + "="*60)
        print("FORECAST VISUALIZATION")
        print("="*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Main forecast plot
        self.model.plot(self.forecast, ax=axes[0, 0])
        axes[0, 0].set_title('Crime Rate Forecast', fontsize=14)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Crime Count')
        axes[0, 0].legend(['Actual', 'Forecast', 'Uncertainty'])
        
        # 2. Trend component
        axes[0, 1].plot(self.forecast['ds'], self.forecast['trend'], linewidth=2)
        axes[0, 1].set_title('Trend Component', fontsize=14)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Trend')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Yearly seasonality
        yearly_seasonality = self.forecast[['ds', 'yearly']].copy()
        yearly_seasonality['month'] = yearly_seasonality['ds'].dt.month
        monthly_avg = yearly_seasonality.groupby('month')['yearly'].mean()
        
        axes[1, 0].bar(range(1, 13), monthly_avg.values)
        axes[1, 0].set_title('Yearly Seasonality Pattern', fontsize=14)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Seasonality Effect')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Weekly seasonality
        weekly_seasonality = self.forecast[['ds', 'weekly']].copy()
        weekly_seasonality['day_of_week'] = weekly_seasonality['ds'].dt.dayofweek
        daily_avg = weekly_seasonality.groupby('day_of_week')['weekly'].mean()
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(days, daily_avg.values)
        axes[1, 1].set_title('Weekly Seasonality Pattern', fontsize=14)
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Seasonality Effect')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Chicago Crime Forecast Analysis', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Also show Prophet's built-in components plot
        print("\nGenerating Prophet components plot...")
        fig2 = self.model.plot_components(self.forecast)
        fig2.suptitle('Prophet Forecast Components', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.forecast is None:
            print("Error: No forecast available.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Calculate metrics on historical data
        historical_forecast = self.forecast[
            self.forecast['ds'] <= self.prophet_df['ds'].max()
        ].copy()
        
        # Merge with actual values
        evaluation_df = pd.merge(
            historical_forecast[['ds', 'yhat']],
            self.prophet_df[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        # Calculate errors
        evaluation_df['error'] = evaluation_df['yhat'] - evaluation_df['y']
        evaluation_df['abs_error'] = abs(evaluation_df['error'])
        evaluation_df['pct_error'] = abs(evaluation_df['error'] / evaluation_df['y']) * 100
        
        # Calculate metrics
        mae = evaluation_df['abs_error'].mean()
        mape = evaluation_df['pct_error'].mean()
        rmse = np.sqrt((evaluation_df['error'] ** 2).mean())
        
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(evaluation_df['ds'], evaluation_df['y'], label='Actual', linewidth=2)
        plt.plot(evaluation_df['ds'], evaluation_df['yhat'], label='Predicted', linewidth=2, alpha=0.8)
        plt.fill_between(evaluation_df['ds'], 
                        evaluation_df['yhat'] - mae, 
                        evaluation_df['yhat'] + mae, 
                        alpha=0.2, color='orange', label='Error Range')
        plt.title('Model Predictions vs Actual Values', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Crime Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'evaluation_df': evaluation_df
        }
    
    def save_results(self, output_dir='results'):
        """Save forecast results and plots"""
        if self.forecast is None:
            print("Error: No forecast available.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forecast data
        forecast_file = f'{output_dir}/chicago_crime_forecast.csv'
        self.forecast.to_csv(forecast_file, index=False)
        print(f"✓ Forecast saved to: {forecast_file}")
        
        # Save model if needed
        model_file = f'{output_dir}/prophet_model.json'
        from prophet.serialize import model_to_json
        
        with open(model_file, 'w') as f:
            f.write(model_to_json(self.model))
        print(f"✓ Model saved to: {model_file}")
        
        return forecast_file


def main():
    """Main execution function"""
    print("="*70)
    print("CHICAGO CRIME FORECASTING USING FACEBOOK PROPHET")
    print("="*70)
    
    # Initialize forecaster
    forecaster = ChicagoCrimeForecaster(data_dir='data')
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    forecaster.load_data()
    
    # Step 2: Explore data
    print("\n[STEP 2] Exploring data...")
    forecaster.explore_data()
    
    # Step 3: Clean data
    print("\n[STEP 3] Cleaning data...")
    forecaster.clean_data()
    
    # Step 4: Analyze crime patterns
    print("\n[STEP 4] Analyzing crime patterns...")
    forecaster.analyze_crime_patterns()
    
    # Step 5: Prepare Prophet data
    print("\n[STEP 5] Preparing Prophet data...")
    forecaster.prepare_prophet_data()
    
    # Step 6: Build Prophet model
    print("\n[STEP 6] Building Prophet model...")
    forecaster.build_prophet_model(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Step 7: Make forecasts
    print("\n[STEP 7] Making forecasts...")
    forecaster.make_forecast(periods=365)  # Forecast 1 year ahead
    
    # Step 8: Visualize results
    print("\n[STEP 8] Visualizing forecasts...")
    forecaster.visualize_forecast()
    
    # Step 9: Evaluate model
    print("\n[STEP 9] Evaluating model...")
    forecaster.evaluate_model()
    
    # Step 10: Save results
    print("\n[STEP 10] Saving results...")
    forecaster.save_results(output_dir='results')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    return forecaster


if __name__ == "__main__":
    # Install required packages
    print("Note: Make sure you have installed the required packages:")
    print("pip install pandas numpy matplotlib seaborn prophet")
    print("\nIf you encounter issues with Prophet, try:")
    print("pip install prophet==1.1.5")
    
    # Run the main pipeline
    try:
        forecaster = main()
        
        # Example: Show top predictions
        print("\nSample Future Predictions:")
        future_predictions = forecaster.forecast[
            forecaster.forecast['ds'] > '2017-12-31'
        ][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)
        print(future_predictions)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all CSV files are in the 'data' directory")
        print("2. Install Prophet: pip install prophet")
        print("3. For Prophet installation issues, try: conda install -c conda-forge prophet")