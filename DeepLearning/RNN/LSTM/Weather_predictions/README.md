# Weather Predictions using LSTM

This repository focuses on predicting weather variables (temperature, pressure, and humidity) using a Long Short-Term Memory (LSTM) neural network. The project involves data preprocessing, visualization, and the creation of an LSTM model for time series forecasting.

## Prerequisites

Make sure to install the required libraries before running the script:

```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn
```

## Project Structure

- **Data**: The dataset used for weather predictions.
- **Models**: Directory to store trained models.
- **Logs**: Directory for TensorBoard logs.

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the script:

   ```bash
   python LSTM_Weather_Predictions.py
   ```

## Dataset

The dataset (`essential_data_synop.csv`) contains weather-related information.

## Data Preprocessing

1. **Removing Useless Columns**: Eliminating non-numeric columns and encoding categorical variables.
2. **Clean the Dataset**: Sorting data by date, interpolating NaN values, and dropping unnecessary columns.
3. **Remove Columns and Rows**: Dropping specific columns and rows with NaN values.

## Visualization

- Visualizing temperature, pressure, and humidity over time for a specific month.

## Parameters

- `scale`: Scaling factor for data.
- `train_prop`: Proportion of data used for training.
- `sequence_len`: Length of input sequences for the LSTM model.
- `batch_size`: Batch size for training.
- `epochs`: Number of training epochs.
- `fit_verbosity`: Verbosity level during model training.

## Model

- LSTM model architecture with one hidden layer.
- TensorBoard and ModelCheckpoint callbacks are used during training.
- Model is compiled using Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metric.

## Results

- Training and validation metrics are visualized.
- Test loss and MAE are displayed.

## Make a Prediction

1. Load the pre-trained model.
2. Get a random sequence from the test data.
3. Make predictions and visualize the results.

Feel free to adapt file paths, parameters, or add more customization based on your needs. If you encounter any issues or have questions, don't hesitate to ask!