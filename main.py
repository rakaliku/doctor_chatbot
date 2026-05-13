import re
import hmac
import hashlib
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from db_config import init_db, seed_sample_data, SessionLocal, Doctor, Vaccine, Appointment
from chatbot import get_chat_reply
import os
import requests

load_dotenv()
port = int(os.environ.get("PORT", 8000))
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")

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


class CreateOrderRequest(BaseModel):
    amount: int
    currency: str = "INR"
    receipt: str | None = None


class VerifyPaymentRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str


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
        "amount_due": getattr(appointment, "amount_due", 0),
        "payment_status": getattr(appointment, "payment_status", "pending"),
        "razorpay_order_id": getattr(appointment, "razorpay_order_id", None),
        "razorpay_payment_id": getattr(appointment, "razorpay_payment_id", None),
    }


@app.post("/chat/")
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    return get_chat_reply(payload.user_input, db, payload.session_id)


@app.get("/api/payment-config")
def get_payment_config():
    if not RAZORPAY_KEY_ID:
        raise HTTPException(status_code=500, detail="Razorpay key id is not configured")
    return {"key_id": RAZORPAY_KEY_ID}


@app.post("/api/create-order")
def create_order(payload: CreateOrderRequest):
    if payload.amount < 100:
        raise HTTPException(status_code=400, detail="Amount must be at least 100 paise")

    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay credentials are not configured")

    order_payload = {
        "amount": payload.amount,
        "currency": payload.currency.upper(),
        "receipt": payload.receipt or f"receipt_{uuid.uuid4().hex[:24]}",
    }

    try:
        response = requests.post(
            "https://api.razorpay.com/v1/orders",
            json=order_payload,
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            timeout=20,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail="Unable to create Razorpay order") from exc

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Razorpay authentication failed")

    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail="Razorpay order creation failed")

    order = response.json()
    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
    }


@app.post("/api/verify-payment")
def verify_payment(payload: VerifyPaymentRequest):
    if not all(
        [
            payload.razorpay_payment_id,
            payload.razorpay_order_id,
            payload.razorpay_signature,
        ]
    ):
        raise HTTPException(status_code=400, detail="Missing payment verification fields")

    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay key secret is not configured")

    message = f"{payload.razorpay_order_id}|{payload.razorpay_payment_id}"
    generated_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(generated_signature, payload.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment signature verification failed")

    return {"success": True, "message": "Payment verified successfully"}


@app.post("/appointments/{appointment_id}/create-order")
def create_order_for_appointment(appointment_id: int, payload: CreateOrderRequest, db: Session = Depends(get_db)):
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if payload.amount < 100:
        raise HTTPException(status_code=400, detail="Amount must be at least 100 paise")

    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay credentials are not configured")

    order_payload = {
        "amount": payload.amount,
        "currency": payload.currency.upper(),
        "receipt": payload.receipt or f"appointment_{appointment_id}_{uuid.uuid4().hex[:12]}",
    }

    try:
        response = requests.post(
            "https://api.razorpay.com/v1/orders",
            json=order_payload,
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            timeout=20,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail="Unable to create Razorpay order") from exc

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Razorpay authentication failed")

    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail="Razorpay order creation failed")

    order = response.json()

    # persist order info on appointment
    appointment.razorpay_order_id = order.get("id")
    appointment.amount_due = payload.amount
    appointment.payment_status = "pending"
    db.add(appointment)
    db.commit()

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
    }


@app.post("/appointments/{appointment_id}/verify-payment")
def verify_payment_for_appointment(appointment_id: int, payload: VerifyPaymentRequest, db: Session = Depends(get_db)):
    if not all(
        [
            payload.razorpay_payment_id,
            payload.razorpay_order_id,
            payload.razorpay_signature,
        ]
    ):
        raise HTTPException(status_code=400, detail="Missing payment verification fields")

    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay key secret is not configured")

    message = f"{payload.razorpay_order_id}|{payload.razorpay_payment_id}"
    generated_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(generated_signature, payload.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment signature verification failed")

    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    # verify order id matches
    if appointment.razorpay_order_id and appointment.razorpay_order_id != payload.razorpay_order_id:
        raise HTTPException(status_code=400, detail="Order id does not match appointment")

    appointment.payment_status = "paid"
    appointment.razorpay_payment_id = payload.razorpay_payment_id
    db.add(appointment)
    db.commit()

    return {"success": True, "message": "Payment verified and linked to appointment"}


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
