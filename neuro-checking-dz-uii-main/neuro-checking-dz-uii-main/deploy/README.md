# Деплой на сервер — check.bruster-go.ru

## Требования к серверу
- Ubuntu 22.04+
- Python 3.11+
- nginx
- git

---

## 1. Клонирование и настройка

```bash
# Клонировать репозиторий
git clone git@gitlab.com:bruster/neuro-checking-dz-uii.git /var/www/neuro-checking
cd /var/www/neuro-checking

# Создать виртуальное окружение и установить зависимости
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Создать .env из шаблона и заполнить
cp .env.example .env
nano .env
```

**Обязательные переменные в `.env` на сервере:**
```
OPENAI_API_KEY=sk-...
FLASK_ENV=production
SESSION_COOKIE_SECURE=true
SECRET_KEY=<случайная строка 64 символа>
PASSWORD=<пароль для входа>
AUTH_EMAIL=<email для входа>
EXTERNAL_SERVICE_URL=http://62.113.108.33/platform-v1/solving-dz
EXTERNAL_SERVICE_AUTH=b59210ae-1493-46c6-b37b-8e89ffa86d90
RUB_PER_TOKEN=0.0005
RUBLES_IN_DOLLAR=75
LOG_LEVEL=INFO
```

Сгенерировать `SECRET_KEY`:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## 2. Systemd-сервис

```bash
sudo cp deploy/neuro-checking.service /etc/systemd/system/
sudo chown -R www-data:www-data /var/www/neuro-checking
sudo systemctl daemon-reload
sudo systemctl enable neuro-checking
sudo systemctl start neuro-checking
sudo systemctl status neuro-checking
```

---

## 3. Nginx

```bash
sudo apt install nginx -y
sudo cp deploy/nginx.conf /etc/nginx/sites-available/neuro-checking
sudo ln -s /etc/nginx/sites-available/neuro-checking /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 4. SSL (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d check.bruster-go.ru
```

После этого certbot автоматически обновит nginx-конфиг с сертификатами.

---

## 5. Обновление (после изменений в git)

```bash
cd /var/www/neuro-checking
git pull
.venv/bin/pip install -r requirements.txt
sudo systemctl restart neuro-checking
```

---

## Диагностика

```bash
# Логи приложения
sudo journalctl -u neuro-checking -f

# Логи nginx
sudo tail -f /var/log/nginx/error.log

# Логи gunicorn
tail -f /var/www/neuro-checking/backend/logs/error.log
```
