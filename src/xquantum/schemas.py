"""Explicit units, unknown values, and source provenance at the data boundary."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Label = Literal[
    "alopecia_areata",
    "androgenetic_alopecia",
    "telogen_effluvium",
    "traction_alopecia",
    "scarring_alopecia",
    "tinea_capitis",
    "scalp_psoriasis",
    "seborrheic_dermatitis",
    "folliculitis",
    "unaffected",
]


OPEN_LICENSES = {
    "Public domain",
    "CC0",
    "CC BY 2.0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 2.0",
    "CC BY-SA 2.5",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
}


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Labs(StrictRecord):
    """Optional results in named units; no automated deficiency interpretation."""

    ferritin_ng_ml: float | None = Field(default=None, ge=0, le=100000)
    hemoglobin_g_dl: float | None = Field(default=None, ge=0, le=30)
    tsh_miu_l: float | None = Field(default=None, ge=0, le=1000)
    vitamin_d_ng_ml: float | None = Field(default=None, ge=0, le=1000)
    zinc_ug_dl: float | None = Field(default=None, ge=0, le=10000)


class ClinicalRecord(StrictRecord):
    """Pseudonymous features only; absent is distinct from a negative finding."""

    age_years: float | None = Field(default=None, ge=0, le=120)
    duration_months: float | None = Field(default=None, ge=0, le=1440)
    sex_at_birth: Literal["female", "male", "intersex", "unknown"] = "unknown"
    skin_phototype: int | None = Field(default=None, ge=1, le=6)
    itching: bool | None = None
    pain: bool | None = None
    scaling: bool | None = None
    pustules: bool | None = None
    sudden_shedding: bool | None = None
    family_history: bool | None = None
    recent_illness: bool | None = None
    postpartum: bool | None = None
    traction_exposure: bool | None = None
    loss_of_follicular_openings: bool | None = None
    miniaturization: bool | None = None
    yellow_dots: bool | None = None
    black_dots: bool | None = None
    broken_hairs: bool | None = None
    perifollicular_erythema: bool | None = None
    labs: Labs = Field(default_factory=Labs)

    def numeric_values(self) -> list[float]:
        values = self.model_dump(exclude={"labs", "sex_at_birth"})
        values.update(self.labs.model_dump())
        values.update(
            {
                f"sex_{sex}": None
                if self.sex_at_birth == "unknown"
                else float(self.sex_at_birth == sex)
                for sex in ("female", "male", "intersex")
            }
        )
        return [float("nan") if value is None else float(value) for value in values.values()]


class Sample(StrictRecord):
    image_path: str = Field(min_length=1, max_length=500)
    label: Label
    patient_id: str | None = Field(default=None, min_length=1, max_length=100)
    source_group: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    license: str = Field(min_length=1, max_length=200)
    rights_basis: str = Field(min_length=1, max_length=2000)
    use_scope: Literal["open", "noncommercial_research", "licensed_research"] = "open"
    permission_reference: str | None = Field(default=None, min_length=1, max_length=1000)
    image_redistribution_allowed: bool = False
    split: Literal["train", "validation", "test"] = "train"
    label_status: Literal["source_only", "specialist_reviewed"] = "source_only"
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=100)
    photo_type: Literal["scalp", "trichoscopy", "unknown"] = "unknown"
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clinical: ClinicalRecord = Field(default_factory=ClinicalRecord)

    @model_validator(mode="after")
    def reviewed_requires_reviewer(self):
        if self.label_status == "specialist_reviewed" and not self.reviewer_id:
            raise ValueError("specialist review requires reviewer_id")
        if self.use_scope == "open" and self.license not in OPEN_LICENSES:
            raise ValueError("Unknown open license; declare restricted use scope and permission")
        if self.use_scope != "open" and not self.permission_reference:
            raise ValueError("Restricted sources require a training permission reference")
        return self


class AnalyzeRequest(StrictRecord):
    image_base64: str = Field(min_length=1, max_length=14_000_000)
    clinical: ClinicalRecord = Field(default_factory=ClinicalRecord)


class AnalyzeCaseRequest(StrictRecord):
    images_base64: list[Annotated[str, Field(min_length=1, max_length=14_000_000)]] = Field(
        min_length=1, max_length=6
    )
    clinical: ClinicalRecord = Field(default_factory=ClinicalRecord)
