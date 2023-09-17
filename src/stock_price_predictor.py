import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


class StockPricePredictor:
    def __init__(self):
        self.input_dim = None
        self.output_dim = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.best_hyperparameters = None

    def load_and_preprocess_data(self, data):
        
        dataset = data[['Close']]
        dataset['Close'] = self.scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))
        return dataset

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length:i + self.sequence_length + 1]
            sequences.append((sequence, target))
            
        return sequences

    def train_val_test_split(self, sequences):
        split_ratio = 0.8
        split_index = int(len(sequences) * split_ratio)

        train_sequences = sequences[:split_index]
        test_sequences = sequences[split_index:]

        X_train, y_train = zip(*train_sequences)
        X_test, y_test = zip(*test_sequences)

        X_train, y_train, X_test, y_test = (
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_data_loaders(self, batch_size=10):
        train_loader = DataLoader(
            dataset=list(zip(self.X_train, self.y_train)),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=list(zip(self.X_val, self.y_val)),
            batch_size=batch_size
        )
        test_loader = DataLoader(
            dataset=list(zip(self.X_test, self.y_test)),
            batch_size=batch_size
        )
        
        return train_loader, val_loader, test_loader


    class LSTMModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_dim=2):
            super(StockPricePredictor.LSTMModel, self).__init__()

            self.hidden_dim = hidden_size
            self.layer_dim = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
            return out

    def create_model(self, input_dim, output_dim, learning_rate, hidden_units, num_lstm_layers, dropout_rate):
        model = self.LSTMModel(input_dim, hidden_units, num_lstm_layers, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        return model, optimizer, criterion

    def tune_hyperparameters(
        self, learning_rates, batch_sizes, hidden_units, num_lstm_layers, dropout_rates, num_epochs=30
    ):
        validation_losses = []

        for lr in learning_rates:
            for bs in batch_sizes:
                for hu in hidden_units:
                    for nl in num_lstm_layers:
                        for dr in dropout_rates:
                            model, optimizer, criterion = self.create_model(
                                self.input_dim, self.output_dim, lr, hu, nl, dr
                            )

                            train_loader, val_loader, _ = self.create_data_loaders(bs)

                            for epoch in range(num_epochs):
                                model.train()
                                for batch_X, batch_y in train_loader:
                                    optimizer.zero_grad()
                                    outputs = model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    optimizer.step()

                                # if (epoch + 1) % 10 == 0:
                                #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                            model.eval()
                            val_loss = 0.0
                            with torch.no_grad():
                                for batch_X, batch_y in val_loader:
                                    val_outputs = model(batch_X)
                                    val_loss += criterion(val_outputs, batch_y).item()

                            average_val_loss = val_loss / len(val_loader)
                            validation_losses.append({
                                'learning_rate': lr,
                                'batch_size': bs,
                                'hidden_units': hu,
                                'num_lstm_layers': nl,
                                'dropout_rate': dr,
                                'validation_loss': average_val_loss
                            })

        sorted_validation_losses = sorted(validation_losses, key=lambda x: x['validation_loss'])
        self.best_hyperparameters = sorted_validation_losses[0]

    def validation_to_csv(self, csv_file_path):
        filename = csv_file_path.split('.')[0]
        output_filename = f'{filename}_validation_losses.csv'
        df = pd.DataFrame(self.validation_losses)
        df.to_csv(output_filename, index=False)

    def evaluate_final_model(self):
        self.model.eval()
        criterion = nn.MSELoss()
        test_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        average_test_loss = test_loss / len(self.test_loader)
        print(f"Average Test Loss: {average_test_loss:.4f}")

    def make_predictions(self):
        last_10_days_data_scaled = self.dataset.iloc[-11:-1].values
        input_data = torch.tensor(last_10_days_data_scaled, dtype=torch.float32).unsqueeze(0)
        original_data = self.data[['Close']].values.reshape(-1, 1)
        self.scaler.fit(original_data)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_data)

        predicted_price = self.scaler.inverse_transform(prediction)
        return predicted_price

    def train(self, data):
        self.data = data
        self.dataset = self.load_and_preprocess_data(data)
        self.sequences = self.create_sequences(self.dataset.values)
        (
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        ) = self.train_val_test_split(self.sequences)
        self.input_dim = self.X_train.shape[-1]
        self.output_dim = 1
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders()
        self.tune_hyperparameters(
            [0.1, 0.01], [32, 64], [64, 128], [1, 2], [0.2, 0.3]
        )
        self.model, self.optimizer, self.criterion = self.create_model(
            self.input_dim, self.output_dim, self.best_hyperparameters['learning_rate'],
            self.best_hyperparameters['hidden_units'], self.best_hyperparameters['num_lstm_layers'],
            self.best_hyperparameters['dropout_rate']
        )
        self.validation_losses = []
        self.evaluate_final_model()
        predictions = self.make_predictions()
        return predictions
