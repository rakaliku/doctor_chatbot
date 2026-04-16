from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db_config import init_db, seed_sample_data, SessionLocal, Doctor, Vaccine, Appointment
from chatbot import get_chat_reply

app = FastAPI()
init_db()
seed_sample_data()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
