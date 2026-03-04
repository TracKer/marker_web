#!/bin/sh
exec uvicorn marker.scripts.server:app --host 0.0.0.0 --port 8000
