from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class MedicalImageCreate(BaseModel):
    filename: str
    modality: Optional[str] = None


class RadiologyReportCreate(BaseModel):
    filename: Optional[str]
    text: Optional[str]


class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    previous_diseases: Optional[str] = None
    symptoms: Optional[str] = None
    notes: Optional[str] = None


class MedicalImageOut(BaseModel):
    id: str
    filename: str
    file_path: str
    modality: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class RadiologyReportOut(BaseModel):
    id: str
    filename: Optional[str]
    text: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class BMDResultOut(BaseModel):
    id: str
    modality: str
    bmd: Optional[str]
    t_score: Optional[str]
    risk_level: Optional[str]
    diagnostic: Optional[str]
    verification: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PatientOut(BaseModel):
    id: str
    name: str
    age: Optional[int]
    gender: Optional[str]
    medical_history: Optional[str]
    previous_diseases: Optional[str]
    symptoms: Optional[str]
    notes: Optional[str]
    created_at: datetime
    images: List[MedicalImageOut] = []
    reports: List[RadiologyReportOut] = []
    bmd_results: List[BMDResultOut] = []

    model_config = ConfigDict(from_attributes=True)
