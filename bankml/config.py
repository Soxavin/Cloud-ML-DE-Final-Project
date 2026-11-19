"""The frozen feature lists (decided in notebook section 4.9, 19 Nov).

Everything that needs to know which columns the model uses imports them from
here: the notebook, the placeholder training script and the tests. Changing
this list changes the API contract, so agree it with the backend side first.
"""

# text columns -> one-hot encoded
CATEGORICAL = ["job", "marital", "education", "default", "contact", "month", "day_of_week", "poutcome"]

# number columns -> standard scaled (previously_contacted is made from pdays by the flag step)
NUMERIC = ["age", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
           "nr.employed", "previously_contacted"]

# long-tailed counts -> log1p, then scaled
LOG_NUMERIC = ["campaign"]

# the raw columns the model takes in, in order; this is what the model service expects
MODEL_INPUTS = ["age", "job", "marital", "education", "default", "contact", "month", "day_of_week",
                "campaign", "pdays", "previous", "poutcome",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
