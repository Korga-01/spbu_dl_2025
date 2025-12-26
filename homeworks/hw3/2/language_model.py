"""
Простая LSTM модель для языкового моделирования
Использует BPE токенизатор
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LSTMLanguageModel(nn.Module):
    """
    Простая LSTM модель для языкового моделирования
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_token_id: int = 1
    ):
        """
        Args:
            vocab_size: Размер словаря токенизатора
            embedding_dim: Размерность эмбеддингов
            hidden_dim: Размерность скрытого состояния LSTM
            num_layers: Количество слоев LSTM
            dropout: Вероятность dropout
            pad_token_id: ID токена заполнения
        """
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Эмбеддинги токенов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Выходной слой
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-init_range, init_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: Токены входной последовательности [batch_size, seq_len]
            hidden: Скрытое состояние LSTM (опционально)
        
        Returns:
            logits: Предсказания для следующего токена [batch_size, seq_len, vocab_size]
            hidden: Новое скрытое состояние
        """
        # Эмбеддинги
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        
        # Выходной слой
        logits = self.output_projection(lstm_out)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden
    
    def generate(
        self,
        tokenizer,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = 'cpu'
    ) -> str:
        """
        Генерирует текст на основе промпта
        
        Args:
            tokenizer: BPE токенизатор
            prompt: Начальный текст
            max_length: Максимальная длина генерируемого текста
            temperature: Температура для сэмплирования (выше = более случайно)
            top_k: Топ-k сэмплирование
            device: Устройство (cpu/cuda)
        
        Returns:
            Сгенерированный текст
        """
        self.eval()
        self.to(device)
        
        # Токенизируем промпт
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        generated = input_ids.clone()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.forward(input_ids, hidden)
                
                # Берем последний токен
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k сэмплирование
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                    # Создаем распределение только для top-k
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                # Проверяем на EOS токен
                if next_token == tokenizer.vocab.get(tokenizer.config.eos_token, 3):
                    break
                
                # Добавляем токен к последовательности
                # generated имеет размерность [1, seq_len], добавляем новый токен
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                generated = torch.cat([generated, next_token_tensor], dim=1)
                
                # Для следующего шага используем только последний токен (или можно использовать последние несколько)
                input_ids = next_token_tensor
        
        # Декодируем
        generated_ids = generated[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids)
        
        return generated_text

