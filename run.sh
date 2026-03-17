cd /root/autodl-tmp/django_website/pyannote_diarization
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8001 --reload