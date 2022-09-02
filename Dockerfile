FROM python:3.7

WORKDIR /app
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-cased'); AutoModel.from_pretrained('bert-base-cased')"
RUN python -m spacy download en_core_web_sm

COPY rstparser rstparser
COPY script script

ENTRYPOINT python -m rstparser.app.api 0.0.0.0 --port 8080 --cpu --hierarchical-type d2e --model-path /models/d2e.t1/*_best_*