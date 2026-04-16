import os
import re
import uuid
from typing import Dict, Optional, Tuple

import requests
from sqlalchemy.orm import Session

from db_config import Appointment, Doctor

DEFAULT_MODELS = [
    model.strip()
    for model in os.getenv(
        "HF_MODELS",
        "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai,Qwen/Qwen2.5-1.5B-Instruct",
    ).split(",")
    if model.strip()
]
HF_TOKEN = os.getenv("HF_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

_sessions: Dict[str, Dict[str, Optional[str]]] = {}


def _get_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Optional[str]]]:
    actual_session_id = session_id or str(uuid.uuid4())
    if actual_session_id not in _sessions:
        _sessions[actual_session_id] = {
            "patient_name": None,
            "doctor_id": None,
            "date_time": None,
            "appointment_id": None,
        }
    return actual_session_id, _sessions[actual_session_id]


def _extract_name(user_input: str) -> Optional[str]:
    patterns = [
        r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"(?:patient name is|patient is|name is)\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"name\s*[:\-]\s*([A-Za-z][A-Za-z\s]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!").title()
    return None


def _extract_doctor(db: Session, user_input: str) -> Optional[Doctor]:
    id_match = re.search(r"(?:doctor|dr|doc)\s*(?:id)?\s*#?\s*(\d+)", user_input, re.IGNORECASE)
    if id_match:
        return db.query(Doctor).filter(Doctor.id == int(id_match.group(1))).first()

    lowered = user_input.lower()
    for doctor in db.query(Doctor).all():
        if doctor.name.lower() in lowered:
            return doctor
        short_name = doctor.name.lower().replace("dr. ", "").replace("dr ", "")
        if short_name in lowered:
            return doctor
    return None


def _extract_requested_doctor_name(user_input: str) -> Optional[str]:
    patterns = [
        r"(?:doctor|dr\.?)\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"(?:appointment of|appointment with|book with)\s+([A-Za-z][A-Za-z\s]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!").title()
    return None


def _extract_date_time(user_input: str) -> Optional[str]:
    patterns = [
        r"\b(\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?)?)\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?)?)\b",
        r"\b(tomorrow(?:\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)?)\b",
        r"\b(today(?:\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _list_doctors(db: Session) -> str:
    doctors = db.query(Doctor).order_by(Doctor.id).all()
    if not doctors:
        return "There are no doctors configured right now."
    return "\n".join(
        f"{doctor.id}. {doctor.name} - {doctor.specialization} ({doctor.available_time})"
        for doctor in doctors
    )


def _build_rule_reply(session: Dict[str, Optional[str]], doctor: Optional[Doctor], booked: bool) -> str:
    if booked and doctor:
        return (
            f"Your dummy appointment is booked. "
            f"Patient: {session['patient_name']}, Doctor: {doctor.name}, "
            f"Time: {session['date_time']}."
        )
    if not session["patient_name"]:
        return "I can help book a dummy doctor appointment. Please share the patient's name."
    if not session["doctor_id"]:
        return (
            "Please choose a doctor by id or name. Available doctors are:\n"
            f"{session.get('doctor_list', '')}"
        )
    if not session["date_time"]:
        return "Please choose the preferred appointment date and time using the picker below."
    return "I have the details I need and can book the appointment now."


def _get_next_step(session: Dict[str, Optional[str]], booked: bool) -> str:
    if booked:
        return "completed"
    if not session["patient_name"]:
        return "patient_name"
    if not session["doctor_id"]:
        return "doctor_id"
    if not session["date_time"]:
        return "date_time"
    return "review"


def _generate_hf_reply(rule_reply: str) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_HF_TOKEN_HERE":
        return None, "fallback", "HF token is missing.", None

    errors = []

    for model_name in DEFAULT_MODELS:
        try:
            response = requests.post(
                HF_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You rewrite hospital chatbot replies to sound short, warm, and clear. "
                                "Do not change the meaning. "
                                "Do not ask for a different field than the one already requested."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Rewrite this message and return only the rewritten message:\n\n{rule_reply}",
                        },
                    ],
                    "max_tokens": 80,
                    "temperature": 0.2,
                },
                timeout=30,
            )
            if not response.ok:
                errors.append(f"{model_name} -> {response.status_code}")
                continue

            data = response.json()
            cleaned = data["choices"][0]["message"]["content"].strip()
            return (cleaned or None), "huggingface", None if cleaned else "Empty HF reply.", model_name
        except Exception as exc:
            errors.append(f"{model_name} -> {exc}")

    return None, "fallback", "HF failed for models: " + " | ".join(errors), None


def get_chat_reply(user_input: str, db: Session, session_id: Optional[str] = None):
    session_id, session = _get_session(session_id)
    doctor_list = _list_doctors(db)

    name = _extract_name(user_input)
    if not name and not session["patient_name"]:
        stripped = user_input.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z\s]{1,40}", stripped):
            lowered = stripped.lower()
            if lowered not in {"show doctors", "list doctors", "available doctors"}:
                name = stripped.title()
    if name:
        session["patient_name"] = name

    doctor = _extract_doctor(db, user_input)
    requested_doctor_name = _extract_requested_doctor_name(user_input)
    if doctor:
        session["doctor_id"] = str(doctor.id)
    elif session.get("doctor_id"):
        doctor = db.query(Doctor).filter(Doctor.id == int(session["doctor_id"])).first()

    date_time = _extract_date_time(user_input)
    if date_time:
        session["date_time"] = date_time

    if requested_doctor_name and not doctor:
        rule_reply = (
            f"Sorry, I could not find a doctor named {requested_doctor_name}. "
            f"Please choose one of the available doctors:\n{doctor_list}"
        )
        hf_reply, response_source, response_error, model_used = _generate_hf_reply(rule_reply)
        return {
            "session_id": session_id,
            "reply": hf_reply or rule_reply,
            "booking_completed": False,
            "appointment_id": None,
            "next_step": "doctor_id",
            "response_source": response_source,
            "model_used": model_used,
            "response_error": response_error,
        }

    wants_doctors = any(keyword in user_input.lower() for keyword in ["doctor", "available", "list"])
    if wants_doctors and not session.get("doctor_id"):
        rule_reply = "Here are the available doctors:\n" + doctor_list
        hf_reply, response_source, response_error, model_used = _generate_hf_reply(rule_reply)
        return {
            "session_id": session_id,
            "reply": hf_reply or rule_reply,
            "booking_completed": False,
            "appointment_id": None,
            "next_step": _get_next_step(session, False),
            "response_source": response_source,
            "model_used": model_used,
            "response_error": response_error,
        }

    booked = False
    appointment_id = None
    if session["patient_name"] and session["doctor_id"] and session["date_time"]:
        appointment = Appointment(
            patient_name=session["patient_name"],
            doctor_id=int(session["doctor_id"]),
            date_time=session["date_time"],
        )
        db.add(appointment)
        db.commit()
        db.refresh(appointment)

        booked = True
        appointment_id = appointment.id
        session["appointment_id"] = str(appointment.id)
        doctor = db.query(Doctor).filter(Doctor.id == int(session["doctor_id"])).first()

    rule_reply = _build_rule_reply({**session, "doctor_list": doctor_list}, doctor, booked)
    hf_reply, response_source, response_error, model_used = _generate_hf_reply(rule_reply)
    reply = hf_reply or rule_reply

    if booked:
        _sessions[session_id] = {
            "patient_name": None,
            "doctor_id": None,
            "date_time": None,
            "appointment_id": None,
        }

    return {
        "session_id": session_id,
        "reply": reply,
        "booking_completed": booked,
        "appointment_id": appointment_id,
        "next_step": _get_next_step(session, booked),
        "response_source": response_source,
        "model_used": model_used,
        "response_error": response_error,
    }
