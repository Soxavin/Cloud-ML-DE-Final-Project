import os

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8001")
MODEL_TIMEOUT_SECONDS = 5.0

# Last month in the dataset (Nov 2010). Dataset column names. Callers cannot override these.
ECONOMIC_CONTEXT = {
    "emp.var.rate": -1.1,
    "cons.price.idx": 94.767,
    "cons.conf.idx": -50.8,
    "euribor3m": 1.028,
    "nr.employed": 4963.6,
}
