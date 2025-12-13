# راهنمای تست وب‌سوکت `/ws`

مراحل زیر کمک می‌کند اتصال، احراز هویت JWT، پیام‌های echo و ping را بررسی کنید.

## پیش‌نیاز

- نصب پیش‌نیازها:

```bash
pip install -r requirements-dev.txt
```

- مقداردهی کلید JWT (اختیاری؛ مقدار پیش‌فرض `secret` است):

```bash
export JWT_SECRET=secret
```

## اجرای سرور محلی

```bash
uvicorn ws_server:app --reload --port 8000
```

### اتصال دستی با websocat

```bash
token=$(python - <<'PY'
from ws_server import create_jwt
print(create_jwt({"sub": "tester"}))
PY)

websocat "ws://127.0.0.1:8000/ws?token=${token}"
```

در ترمینال websocat، پیام زیر را بفرستید و پاسخ را مشاهده کنید:

```json
{"type": "delta", "delta": "سلام"}
```

انتظار می‌رود پاسخ زیر دریافت شود و هر ۲۰ ثانیه یک پیام ping دریافت کنید:

```json
{"type": "delta", "delta": "سلام", "echoed": true}
{"type": "ping"}
```

## اجرای تست واحد با pytest-asyncio

```bash
pytest tests/test_ws_server.py -k ping -q
```

این تست با کوتاه کردن بازه ping به 0.01 ثانیه، پیام echo و یک ping را دریافت و بررسی می‌کند.
