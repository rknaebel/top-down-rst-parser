FROM python:3.8

WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-cased'); AutoModel.from_pretrained('bert-base-cased')"
RUN python -m nltk.downloader maxent_treebank_pos_tagger punkt

COPY data data
COPY rstparser rstparser
COPY script script

ENTRYPOINT bash