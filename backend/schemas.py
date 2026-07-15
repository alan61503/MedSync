from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class MedicalImageCreate(BaseModel):
    filename: str
    modality: Optional[str]


class RadiologyReportCreate(BaseModel):
    filename: Optional[str]
    text: Optional[str]


class PatientCreate(BaseModel):
    name: str
    age: Optional[int]
    gender: Optional[str]
    medical_history: Optional[str]
    previous_diseases: Optional[str]
    symptoms: Optional[str]
    notes: Optional[str]


class MedicalImageOut(BaseModel):
    id: str
    filename: str
    file_path: str
    modality: Optional[str]

    class Config:
        orm_mode = True


class RadiologyReportOut(BaseModel):
    id: str
    filename: Optional[str]
    text: Optional[str]

    class Config:
        orm_mode = True


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

    class Config:
        orm_mode = True
