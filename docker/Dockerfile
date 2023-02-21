FROM python:3.9-buster

RUN pip install --upgrade pip
RUN pip install pandas==1.4.2 sklearn pyarrow joblib s3fs lightgbm 
ENV PYTHONUNBUFFERED=TRUE  
ENV PYTHONPATH "${PYTHONPATH}:/opt/ml/processing/input/"

ENTRYPOINT ["python3"]
