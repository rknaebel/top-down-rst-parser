FROM python:3.7

RUN pip install -U pip setuptools wheel

WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -c "import torchtext.vocab; torchtext.vocab.GloVe()"
RUN python -m nltk.downloader maxent_treebank_pos_tagger punkt

COPY data data
COPY rstparser rstparser
COPY script script

ENTRYPOINT bash