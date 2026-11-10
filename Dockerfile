FROM python:3.10-slim

WORKDIR /app

# Only the packages needed to serve the model
RUN pip install --no-cache-dir \
    pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 xgboost==2.1.2 \
    flask==3.0.3 gunicorn==23.0.0

COPY features.py predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "--workers=2", "predict:app"]
