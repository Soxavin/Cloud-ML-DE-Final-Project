from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Job = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
Default = Literal["no", "yes", "unknown"]
Contact = Literal["cellular", "telephone"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["failure", "nonexistent", "success"]


class PredictRequest(BaseModel):
    # Unknown / misspelled fields give 422 instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    age: Annotated[int, Field(ge=17, le=100)]
    job: Job
    marital: Marital
    education: Education
    default: Default
    contact: Contact
    month: Month
    day_of_week: DayOfWeek
    campaign: Annotated[int, Field(ge=1)]
    # Required, but may be null (= never contacted before).
    days_since_last_contact: Optional[Annotated[int, Field(ge=0, le=30)]]
    previous: Annotated[int, Field(ge=0)]
    poutcome: Poutcome
    project: Optional[str] = None  # label for screenshots, never sent to the model


class PredictResponse(BaseModel):
    prediction: Literal["yes", "no"]
    subscribe_probability: float
    threshold: float
    model_version: str
    economic_context: dict[str, float]
    input_received: dict
