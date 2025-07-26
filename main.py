# --- Импорты и настройка среды ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# --- Конфигурация ---
CONFIG = {
    'BATCH_SIZE': 1024,
    'NUM_WORKERS': 0,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 1e-5,
    'TEST_SIZE': 0.3,
    'VAL_SIZE': 0.33,  # 2/3 от test_size
    'RANDOM_STATE': 42
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device.type.upper()}")


# --- Класс модели ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers[:-2])  # Remove last ReLU
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1]))  # Exclude bottleneck
        prev_dim = hidden_dims[-1]  # Bottleneck dimension
        for dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # Output layer
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# --- Функции обработки данных ---
def load_and_preprocess_data(filepath):
    """Загрузка и предварительная обработка данных"""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Данные загружены. Размер: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{filepath}' не найден.")
    
    # Очистка данных
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Timestamp']).copy()
    
    return df

def create_features(df):
    """Создание признаков"""
    # Временные признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24.0)
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    
    # Частотные признаки
    df['source_ip_freq'] = df['Source IP'].map(df['Source IP'].value_counts(normalize=True))
    df['dest_ip_freq'] = df['Destination IP'].map(df['Destination IP'].value_counts(normalize=True))
    df['dest_port_is_well_known'] = (df['Destination Port'] < 1024).astype(int)
    
    return df

def prepare_features_and_target(df):
    """Подготовка признаков и целевой переменной"""
    columns_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                      'Destination IP', 'Timestamp', 'Class']
    
    df_processed = df.drop(columns=columns_to_drop, errors='ignore')
    df_processed['Class_ID'] = (df['Class'] == 'Keylogger').astype(int)
    
    X = df_processed.select_dtypes(include=np.number).drop(columns=['Class_ID'])
    y = df_processed['Class_ID']
    
    # Обработка бесконечных значений
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Инжиниринг завершен. Количество признаков: {X.shape[1]}")
    
    return X, y

def split_and_scale_data(X, y):
    """Разделение и масштабирование данных"""
    # Разделение на классы
    X_benign = X[y == 0]
    X_keylogger = X[y == 1]
    
    # Разделение "хороших" данных
    X_train, X_temp = train_test_split(
        X_benign, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    X_val, X_test_benign = train_test_split(
        X_temp, 
        test_size=CONFIG['VAL_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_benign_scaled = scaler.transform(X_test_benign)
    X_keylogger_scaled = scaler.transform(X_keylogger)
    
    # Финальные выборки
    X_test = np.concatenate([X_test_benign_scaled, X_keylogger_scaled])
    y_test = np.concatenate([
        np.zeros(len(X_test_benign_scaled)), 
        np.ones(len(X_keylogger_scaled))
    ])
    
    return {
        'train': X_train_scaled,
        'val': X_val_scaled,
        'test': X_test,
        'test_benign': X_test_benign_scaled,
        'keylogger': X_keylogger_scaled,
        'y_test': y_test,
        'scaler': scaler
    }

# --- Функции обучения ---
def train_model(model, train_loader, X_val_tensor, criterion, optimizer, grad_scaler):
    """Обучение модели"""
    train_losses = []
    val_losses = []
    
    print("\nНачало обучения модели...")
    for epoch in range(CONFIG['NUM_EPOCHS']):
        # Обучение
        model.train()
        train_loss_acc = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_acc += loss.item()
        
        avg_train_loss = train_loss_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, X_val_tensor)
            val_losses.append(val_loss.item())
        
        print(f"Эпоха {epoch+1}/{CONFIG['NUM_EPOCHS']} | "
              f"Ошибка обучения: {avg_train_loss:.6f} | "
              f"Ошибка валидации: {val_loss.item():.6f}")
    
    print("Обучение завершено.\n")
    return train_losses, val_losses

# --- Функции оценки ---
def evaluate_model(model, X_test_tensor, y_test):
    """Оценка модели и поиск оптимального порога"""
    model.eval()
    with torch.no_grad():
        test_reconstructions = model(X_test_tensor)
        test_errors = nn.functional.mse_loss(
            test_reconstructions, X_test_tensor, 
            reduction='none'
        ).mean(dim=1).cpu().numpy()
    
    # Поиск оптимального порога
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_errors)
    roc_auc = auc(fpr, tpr)
    j_statistic = tpr - fpr
    best_j_idx = np.argmax(j_statistic)
    optimal_threshold = roc_thresholds[best_j_idx]
    
    # Предсказания
    y_pred_optimal = (test_errors > optimal_threshold).astype(int)
    
    return {
        'errors': test_errors,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'y_pred': y_pred_optimal,
        'best_j_idx': best_j_idx,
        'j_statistic': j_statistic
    }

def plot_results(results, y_test):
    """Визуализация результатов"""
    # ROC-кривая
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
             label=f'ROC кривая (AUC = {results["roc_auc"]:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Случайное угадывание')
    plt.scatter(results['fpr'][results['best_j_idx']], 
                results['tpr'][results['best_j_idx']], 
                marker='o', color='red', s=100, zorder=5, 
                label=f'Оптимальный порог ({results["optimal_threshold"]:.4f})')
    plt.xlabel('Доля ложных тревог')
    plt.ylabel('Доля верных обнаружений')
    plt.title('ROC-кривая для детектора аномалий')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Матрица ошибок
    plt.subplot(1, 2, 2)
    cm_optimal = confusion_matrix(y_test, results['y_pred'])
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Keylogger'], 
                yticklabels=['Benign', 'Keylogger'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанная метка')
    plt.ylabel('Истинная метка')
    
    plt.tight_layout()
    plt.show()

# --- Основной код ---
def main():
    # Загрузка и обработка данных
    df = load_and_preprocess_data('Keylogger_Detection.csv')
    df = create_features(df)
    X, y = prepare_features_and_target(df)
    
    # Разделение и масштабирование
    data_splits = split_and_scale_data(X, y)
    print(f"Размер обучающей выборки: {data_splits['train'].shape}")
    print(f"Размер тестовой выборки: {data_splits['test'].shape}")
    
    # Подготовка модели
    input_dim = data_splits['train'].shape[1]
    model = Autoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), 
                          lr=CONFIG['LEARNING_RATE'], 
                          weight_decay=CONFIG['WEIGHT_DECAY'])
    criterion = nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # DataLoader
    X_train_tensor = torch.FloatTensor(data_splits['train'])
    X_val_tensor = torch.FloatTensor(data_splits['val']).to(device)
    X_test_tensor = torch.FloatTensor(data_splits['test']).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, X_train_tensor), 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    # Обучение
    train_losses, val_losses = train_model(
        model, train_loader, X_val_tensor, 
        criterion, optimizer, grad_scaler
    )
    
    # Оценка
    results = evaluate_model(model, X_test_tensor, data_splits['y_test'])
    print(f"Оптимальный порог: {results['optimal_threshold']:.6f}")
    
    print("\nОтчет по классификации:")
    print(classification_report(data_splits['y_test'], results['y_pred'], 
                              target_names=['Benign', 'Keylogger']))
    
    # Визуализация
    plot_results(results, data_splits['y_test'])
    
    # Сохранение
    artifacts_to_save = {
        'model_state_dict': model.state_dict(),
        'scaler': data_splits['scaler'],
        'threshold': results['optimal_threshold'],
        'input_dim': input_dim,
        'config': CONFIG
    }
    save_path = 'keylogger_detector_artifacts.pth'
    torch.save(artifacts_to_save, save_path)
    print(f"\nМодель и артефакты сохранены в файл: {save_path}")

# Запуск
if __name__ == "__main__":
    main()
