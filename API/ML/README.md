# Command for run a server

```
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

```
ssh -R group6-ml:80:localhost:8000 cn240@ondev.link -i ./cn240.key
```