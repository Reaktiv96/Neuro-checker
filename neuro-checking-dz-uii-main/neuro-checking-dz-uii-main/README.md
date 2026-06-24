# Neuro Checking - AI Homework Assessment System

Веб-приложение для автоматической проверки и генерации вариантов домашних заданий с использованием AI.

## Структура проекта

```
backend/
├── app.py                       # Flask приложение и маршруты
├── config.py                    # Конфигурация
├── services/                    # Основная логика
│   ├── colab_loader.py         # Загрузка из Google Colab
│   ├── code_cleaner.py         # Очистка кода от метаданных
│   ├── precheck.py             # Детерминированная проверка
│   ├── code_parser.py          # Парсинг и структурирование кода
│   ├── logger.py               # Логирование операций
│   └── ...                     # (генератор, оценка - далее)
├── models/                      # Модели данных и промпты
│   ├── __init__.py             # Pydantic схемы
│   └── prompts.py              # Промпты для LLM
├── utils/                       # Утилиты
│   ├── syntax_validator.py     # Валидация синтаксиса Python
│   ├── patch_applier.py        # Применение патчей к коду
│   └── ...
└── logs/                        # Директория с логами

```

## Установка

### 1. Клонируем и переходим в проект
```bash
cd /Users/artemkashtalap/My\ Projects/neuro_checking
```

### 2. Создаем виртуальное окружение (опционально, но рекомендуется)
```bash
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Устанавливаем зависимости
```bash
pip install -r requirements.txt
```

### 4. Создаем .env файл
```bash
cp .env.example .env
# Редактируем .env и добавляем OpenAI API ключ
```

Содержимое `.env`:
```
OPENAI_API_KEY=sk-your-api-key-here
RUB_PER_TOKEN=0.0005
EXTERNAL_SERVICE_URL=http://62.113.108.33/platform-v1/solving-dz
EXTERNAL_SERVICE_AUTH=b59210ae-1493-46c6-b37b-8e89ffa86d90
LOG_LEVEL=INFO
```

## Запуск

```bash
python run.py
```

Приложение запустится на `http://localhost:5000`

## API Endpoints

### ✅ Health Check
```
GET /api/health
```
Проверка статуса приложения.

### 📥 Fetch Colab
```
POST /api/fetch-colab
Content-Type: application/json

{
    "url": "https://colab.research.google.com/drive/..."
}
```

Ответ:
```json
{
    "content": "clean python code",
    "raw_code": "original code with metadata",
    "precheck": {
        "has_valid_code": true,
        "has_syntax_errors": false,
        "has_stubs": false,
        "has_metadata": false,
        "forced_score": null,
        "reasons": []
    },
    "parsed": {
        "executable_code": "...",
        "logs": "...",
        "metadata": {...}
    }
}
```

### 📊 Get Logs
```
GET /api/logs
```
Получить логи использования (для биллинга).

```
GET /api/detailed-logs
```
Список всех детальных логов операций.

```
GET /api/detailed-logs/<filename>
```
Содержимое конкретного лога.

## Фазы реализации

### ✅ Фаза 1: Инфраструктура (завершена)
- [x] Структура проекта
- [x] Конфигурация
- [x] Requirements

### ✅ Фаза 2: Ядро (завершена)
- [x] Code Cleaner - очистка от метаданных
- [x] Colab Loader - загрузка из Colab
- [x] Precheck - детерминированная проверка
- [x] Code Parser - парсинг и структурирование
- [x] Logger - логирование операций
- [x] Syntax Validator - валидация синтаксиса
- [x] Patch Applier - применение патчей
- [x] Базовый API

### ⏳ Фаза 3: Генерация (следующая)
- [ ] Generator Service - генерация корректных вариантов
- [ ] Incorrect Generator - генерация ошибочных вариантов
- [ ] Retry Logic - переподпрос при ошибке

### ⏳ Фаза 4: Оценка
- [ ] Evaluator Service - LLM-проверка
- [ ] Score Forcing - применение forced_score

### ⏳ Фаза 5: Фронтенд
- [ ] HTML интерфейс для тестирования

## Использованные сервисы

- **Google Colab**: источник исходных данных
- **OpenAI GPT-4o**: генерация и оценка
- **External Service**: fallback для загрузки (62.113.108.33)

## Разработка

### Логирование

Все операции логируются в `backend/logs/`:
- `usage_logs.json` - краткий лог использования (API, токены, биллинг)
- `*.txt` - детальные логи каждой операции

### Структура Core

Функционал разбит на независимые сервисы в `backend/services/`:
- Каждый сервис - отдельный модуль
- Можно легко вытащить и внедрить в другой проект
- Минимальные зависимости

## Notes

- Коды ошибок: `syntax_error`, `logical_error`, `non_optimal`, `partial`, `cheating`
- Python версия: 3.9+
- Языки: Python (backend), JavaScript (frontend - далее)
