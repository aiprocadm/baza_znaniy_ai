# SSL и Basic Auth для dev-окружения

В репозитории намеренно отсутствуют приватные ключи и пароли. Сгенерируйте их
перед запуском nginx-контейнера:

```bash
mkdir -p data/ssl
# self-signed сертификат
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -subj "/CN=kb.local" \
  -keyout data/ssl/kb.local.key \
  -out data/ssl/kb.local.crt

# basic auth пароль
htpasswd -bc data/ssl/.htpasswd admin strong-password
```

Не храните реальные секреты в git и используйте `.gitignore`, чтобы не
закоммитить эти файлы случайно.
