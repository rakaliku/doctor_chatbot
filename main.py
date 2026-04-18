import re
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db_config import init_db, seed_sample_data, SessionLocal, Doctor, Vaccine, Appointment
from chatbot import get_chat_reply
import os

port = int(os.environ.get("PORT", 8000))

app = FastAPI()
init_db()
seed_sample_data()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _parse_requested_time(date_time: str):
    normalized = re.sub(r"\s+", " ", date_time.strip())
    formats = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
        "%I:%M %p",
        "%H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _is_within_doctor_hours(doctor: Doctor, requested_date_time: str) -> bool:
    requested = _parse_requested_time(requested_date_time)
    if requested is None:
        return True

    start_text, end_text = [part.strip() for part in doctor.available_time.split("-", 1)]
    start = datetime.strptime(start_text, "%H:%M")
    end = datetime.strptime(end_text, "%H:%M")
    requested_minutes = requested.hour * 60 + requested.minute
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    return start_minutes <= requested_minutes <= end_minutes


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AppointmentRequest(BaseModel):
    patient_name: str
    doctor_id: int
    date_time: str


class ChatRequest(BaseModel):
    user_input: str
    session_id: str | None = None


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/doctors")
def get_doctors(db: Session = Depends(get_db)):
    doctors = db.query(Doctor).all()
    return [{"id": d.id, "name": d.name, "specialization": d.specialization, "available_time": d.available_time} for d in doctors]


@app.get("/vaccines")
def get_vaccines(db: Session = Depends(get_db)):
    vaccines = db.query(Vaccine).all()
    return [{"id": v.id, "name": v.name, "stock": v.stock} for v in vaccines]


@app.post("/book")
def book_appointment(payload: AppointmentRequest, db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.id == payload.doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if not _is_within_doctor_hours(doctor, payload.date_time):
        raise HTTPException(
            status_code=400,
            detail=f"{doctor.name} is available only during {doctor.available_time}",
        )

    appt = Appointment(
        patient_name=payload.patient_name,
        doctor_id=payload.doctor_id,
        date_time=payload.date_time,
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return {"message": "Appointment booked successfully!", "appointment_id": appt.id}


@app.get("/appointments/{appointment_id}")
def get_appointment_status(appointment_id: int, db: Session = Depends(get_db)):
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    doctor = db.query(Doctor).filter(Doctor.id == appointment.doctor_id).first()
    return {
        "appointment_id": appointment.id,
        "status": "booked",
        "patient_name": appointment.patient_name,
        "doctor_name": doctor.name if doctor else None,
        "date_time": appointment.date_time,
    }


@app.post("/chat/")
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    return get_chat_reply(payload.user_input, db, payload.session_id)

@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
