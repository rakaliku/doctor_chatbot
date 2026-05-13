import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db_config import Appointment, Doctor

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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

_sessions: Dict[str, Dict[str, Any]] = {}
_next_session_id = 1


def _get_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    global _next_session_id
    # Numeric session ids avoid confusing chat ids with appointment ids in the UI.
    actual_session_id = session_id
    if not actual_session_id:
        actual_session_id = str(_next_session_id)
        _next_session_id += 1
    if actual_session_id not in _sessions:
        _sessions[actual_session_id] = {
            "patient_name": None,
            "doctor_id": None,
            "date_time": None,
            "appointment_id": None,
            "pending_doctor_id": None,
            "pending_date_time": None,
            "pending_confirmation": None,
            "history": [],
        }
    return actual_session_id, _sessions[actual_session_id]


def _extract_name(user_input: str) -> Optional[str]:
    patterns = [
        r"(?:it's|its|it is)\s+([A-Za-z][A-Za-z\s]{0,40})",
        r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{0,40})",
        r"(?:patient name is|patient is|name is)\s+([A-Za-z][A-Za-z\s]{0,40})",
        r"name\s*[:\-]\s*([A-Za-z][A-Za-z\s]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            candidate_words = match.group(1).strip(" .,!").split()
            stop_words = {
                "and",
                "is",
                "for",
                "to",
                "with",
                "looking",
                "want",
                "need",
                "book",
                "appointment",
                "doctor",
                "dr",
                "tomorrow",
                "today",
            }
            cleaned_words = []
            for word in candidate_words:
                if word.lower() in stop_words:
                    break
                cleaned_words.append(word)
                if len(cleaned_words) == 3:
                    break
            if cleaned_words:
                return " ".join(cleaned_words).title()
    return None


def _looks_like_plain_name(user_input: str) -> bool:
    stripped = user_input.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z\s]{1,40}", stripped):
        return False

    lowered = stripped.lower()
    if len(lowered.split()) > 3:
        return False

    blocked_terms = {
        "appointment",
        "book",
        "doctor",
        "doctors",
        "available",
        "list",
        "show",
        "need",
        "help",
        "want",
        "schedule",
        "hello",
        "hi",
        "hey",
        "how",
        "are",
        "you",
        "your",
        "thanks",
        "thank",
        "fine",
        "good",
        "morning",
        "afternoon",
        "evening",
    }
    words = [word for word in lowered.split() if word]
    if not 1 <= len(words) <= 3:
        return False
    return not any(word in blocked_terms for word in words)


def _extract_doctor(db: Session, user_input: str) -> Optional[Doctor]:
    id_match = re.search(r"(?:doctor|dr|doc)\s*(?:id)?\s*#?\s*(\d+)", user_input, re.IGNORECASE)
    if id_match:
        return db.query(Doctor).filter(Doctor.id == int(id_match.group(1))).first()

    lowered = user_input.lower()
    stripped = lowered.strip()
    has_doctor_context = any(
        phrase in lowered
        for phrase in ["doctor", "dr ", "dr.", "appointment with", "appointment of", "appointment for", "book with", "consult with"]
    )
    doctors = db.query(Doctor).all()

    if has_doctor_context:
        name_match = re.search(
            r"(?:doctor|dr\.?|doc|appointment with|appointment of|appointment for|book with|consult with)\s+([A-Za-z][A-Za-z\s]{0,40})",
            user_input,
            re.IGNORECASE,
        )
        if name_match:
            requested_fragment = name_match.group(1).strip(" .,!").lower()
            # trim off trailing scheduling words
            requested_fragment = re.split(r"\b(?:for|tomorrow|today|at|on|in)\b", requested_fragment)[0].strip()
            # remove leading honorifics like Dr
            requested_fragment = re.sub(r"^dr\.?\s*", "", requested_fragment, flags=re.IGNORECASE).strip()
            fragment_tokens = [
                token for token in re.findall(r"[a-z]+", requested_fragment)
                if token not in {"doctor", "dr", "doc", "appointment", "with", "book", "consult"}
            ]
            if fragment_tokens:
                matched_doctors = []
                for doctor in doctors:
                    doctor_tokens = [
                        token for token in re.findall(r"[a-z]+", doctor.name.lower())
                        if token not in {"dr"}
                    ]
                    if all(token in doctor_tokens for token in fragment_tokens):
                        matched_doctors.append(doctor)
                if len(matched_doctors) == 1:
                    return matched_doctors[0]

    for doctor in doctors:
        if doctor.name.lower() in lowered:
            return doctor

        short_name = doctor.name.lower().replace("dr. ", "").replace("dr ", "")
        if has_doctor_context and short_name in lowered:
            return doctor

        if stripped == short_name:
            return doctor

        if has_doctor_context:
            doctor_tokens = [token for token in re.findall(r"[a-z]+", short_name) if token not in {"dr"}]
            mentioned_tokens = [token for token in re.findall(r"[a-z]+", lowered) if token not in {"doctor", "dr", "doc"}]
            if len(mentioned_tokens) >= 1 and all(token in doctor_tokens for token in mentioned_tokens):
                return doctor
    return None


def _extract_doctor_id_only(db: Session, user_input: str) -> Optional[Doctor]:
    stripped = user_input.strip()
    if not re.fullmatch(r"\d+", stripped):
        return None
    return db.query(Doctor).filter(Doctor.id == int(stripped)).first()


def _extract_requested_doctor_name(user_input: str) -> Optional[str]:
    patterns = [
        r"(?:doctor|dr\.? )\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"(?:appointment of|appointment with|book with|appointment for)\s+([A-Za-z][A-Za-z\s]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            fragment = match.group(1).strip(" .,!").lower()
            # Trim scheduling words and honorifics
            fragment = re.split(r"\b(?:for|tomorrow|today|at|on|in)\b", fragment)[0].strip()
            fragment = re.sub(r"^dr\.?\s*", "", fragment, flags=re.IGNORECASE).strip()
            return fragment.title()
    return None


def _is_doctor_listing_request(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(keyword in lowered for keyword in ["show doctors", "list doctors", "available doctors"])


def _extract_date_time(user_input: str) -> Optional[str]:
    patterns = [
        r"\b(\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?)?)\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?)?)\b",
        r"\b(tomorrow\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\b",
        r"\b(today\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_relative_preference(user_input: str) -> Optional[str]:
    lowered = user_input.lower()
    if "tomorrow evening" in lowered:
        return "tomorrow_evening"
    if "tomorrow afternoon" in lowered:
        return "tomorrow_afternoon"
    if "tomorrow morning" in lowered:
        return "tomorrow_morning"
    if "tomorrow" in lowered:
        return "tomorrow"
    if "today evening" in lowered:
        return "today_evening"
    if "today afternoon" in lowered:
        return "today_afternoon"
    if "today morning" in lowered:
        return "today_morning"
    return None


def _format_time_12h(value: str) -> str:
    parsed = datetime.strptime(value, "%H:%M")
    return parsed.strftime("%I:%M %p").lstrip("0")


def _parse_available_range(available_time: str) -> Tuple[str, str]:
    start_time, end_time = [part.strip() for part in available_time.split("-", 1)]
    return start_time, end_time


def _parse_time_from_text(date_time: str) -> Optional[datetime]:
    normalized = re.sub(r"\s+", " ", date_time.strip())
    formats = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%I:%M %p",
        "%H:%M",
    ]
    prefixes = ("tomorrow at ", "today at ")
    lowered = normalized.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _is_within_doctor_hours(doctor: Doctor, requested_date_time: str) -> bool:
    requested = _parse_time_from_text(requested_date_time)
    if requested is None:
        return True

    start_time, end_time = _parse_available_range(doctor.available_time)
    start_dt = datetime.strptime(start_time, "%H:%M")
    end_dt = datetime.strptime(end_time, "%H:%M")
    requested_minutes = requested.hour * 60 + requested.minute
    start_minutes = start_dt.hour * 60 + start_dt.minute
    end_minutes = end_dt.hour * 60 + end_dt.minute
    return start_minutes <= requested_minutes <= end_minutes


def _build_out_of_hours_reply(doctor: Doctor) -> str:
    start_time, end_time = _parse_available_range(doctor.available_time)
    return (
        f"{doctor.name} is only available between {_format_time_12h(start_time)} and "
        f"{_format_time_12h(end_time)}. Please choose a time within that range."
    )


def _period_matches(available_time: str, preference: str) -> bool:
    start_time, end_time = _parse_available_range(available_time)
    start_hour = int(start_time.split(":")[0])
    end_hour = int(end_time.split(":")[0])

    if preference.endswith("morning"):
        return start_hour < 12
    if preference.endswith("afternoon"):
        return end_hour >= 12
    if preference.endswith("evening"):
        return end_hour >= 17
    return True


def _build_suggested_slot(preference: str, available_time: str) -> str:
    day_label = "tomorrow" if preference.startswith("tomorrow") else "today"
    start_time, _ = _parse_available_range(available_time)
    return f"{day_label} at {_format_time_12h(start_time)}"


def _build_availability_reply(doctor: Doctor, preference: str, session: Dict[str, Any]) -> str:
    start_time, end_time = _parse_available_range(doctor.available_time)
    day_label = "tomorrow" if preference.startswith("tomorrow") else "today"
    start_label = _format_time_12h(start_time)
    end_label = _format_time_12h(end_time)
    suggested_slot = session["pending_date_time"]
    patient_text = f" for {session['patient_name']}" if session.get("patient_name") else ""

    if _period_matches(doctor.available_time, preference):
        return (
            f"{doctor.name} is available {day_label} from {start_label} to {end_label}{patient_text}. "
            f"Would you like me to book the appointment for {suggested_slot}?"
        )

    requested_period = preference.split("_", 1)[1]
    return (
        f"{doctor.name} is available {day_label} from {start_label} to {end_label}, not in the {requested_period}{patient_text}. "
        f"Would you like me to book the appointment for {suggested_slot} instead?"
    )


def _is_confirmation(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    return lowered in {"yes", "y", "ok", "okay", "sure", "proceed", "confirm", "yes please"}


def _is_rejection(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    return lowered in {"no", "n", "nope", "cancel", "not now"}


def _list_doctors(db: Session) -> str:
    doctors = db.query(Doctor).order_by(Doctor.id).all()
    if not doctors:
        return "There are no doctors configured right now."
    return "\n".join(
        f"{doctor.id}. {doctor.name} - {doctor.specialization} ({doctor.available_time})"
        for doctor in doctors
    )


def _build_rule_reply(session: Dict[str, Any], doctor: Optional[Doctor], booked: bool) -> str:
    if booked and doctor:
        return (
            f"Your appointment is booked. "
            f"Patient: {session['patient_name']}, Doctor: {doctor.name}, "
            f"Time: {session['date_time']}. "
            f"Appointment ID: {session['appointment_id']}."
        )
    if not session["patient_name"]:
        return "I can help book a doctor appointment. Please share the patient's name."
    if session.get("pending_confirmation"):
        return "Please reply yes to confirm the suggested slot, or share another preferred time."
    if not session["doctor_id"]:
        return (
            "Please choose a doctor by id or name. Available doctors are:\n"

            f"{session.get('doctor_list', '')}"
        )
    if not session["date_time"]:
        return "Please choose the preferred appointment date and time using the picker below."
    return "I have the details I need and can book the appointment now."


def _ensure_booking_details_in_reply(
    reply: str,
    session: Dict[str, Any],
    doctor: Optional[Doctor],
    appointment_id: Optional[int],
    booked: bool,
) -> str:
    if not booked or not doctor or not appointment_id:
        return reply

    reply = re.sub(
        r"(?im)Appointment\s+ID\s*:.*$",
        f"Appointment ID: {int(appointment_id)}",
        reply,
    ).rstrip()
    required_lines = [
        f"Patient: {session['patient_name']}",
        f"Doctor: {doctor.name}",
        f"Date & Time: {session['date_time']}",
        f"Appointment ID: {int(appointment_id)}",
    ]
    missing_lines = [line for line in required_lines if line not in reply]
    if not missing_lines:
        return reply
    return reply.rstrip() + "\n" + "\n".join(missing_lines)


def _get_next_step(session: Dict[str, Any], booked: bool) -> str:
    if booked:
        return "completed"
    if not session["patient_name"]:
        return "patient_name"
    if not session["doctor_id"]:
        return "doctor_id"
    if not session["date_time"]:
        return "date_time"
    return "review"


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No earlier messages."

    trimmed = history[-8:]
    return "\n".join(f"{item['role']}: {item['content']}" for item in trimmed)


def _build_llm_messages(
    user_input: str,
    rule_reply: str,
    session: Dict[str, Any],
    doctor_list: str,
    next_step: str,
) -> List[Dict[str, str]]:
    patient_name = session.get("patient_name") or "missing"
    doctor_id = session.get("doctor_id") or "missing"
    date_time = session.get("date_time") or "missing"
    appointment_id = session.get("appointment_id") or "not booked"
    history_text = _format_history(session.get("history", []))

    return [
        {
            "role": "system",
            "content": (
                "You are a hospital appointment booking assistant. "
                "Keep replies short, warm, and practical. "
                "Stay grounded in the provided doctor list and booking state. "
                "Never invent doctors, timings, appointment ids, or policies. "
                "Never say an appointment is booked unless next_step is completed. "
                "If a field is missing, ask only for the next missing field. "
                "If the backend provides an appointment id, you must include it in the confirmation. "
                "If the user asks for available doctors, summarize the list clearly. "
                "If the booking is complete, confirm it naturally using the provided facts only. "
                "If the user asks something outside appointment booking, politely steer them back."
            ),
        },
        {
            "role": "system",
            "content": (
                f"Conversation history:\n{history_text}\n\n"
                f"Current booking state:\n"
                f"- patient_name: {patient_name}\n"
                f"- doctor_id: {doctor_id}\n"
                f"- date_time: {date_time}\n"
                f"- appointment_id: {appointment_id}\n"
                f"- pending_confirmation: {session.get('pending_confirmation') or 'none'}\n"
                f"- next_step: {next_step}\n\n"
                f"Available doctors:\n{doctor_list}\n\n"
                f"Reliable backend guidance:\n{rule_reply}"
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]


def _reply_claims_booking(reply: str) -> bool:
    lowered = reply.lower()
    return any(
        phrase in lowered
        for phrase in [
            "appointment is booked",
            "appointment has been booked",
            "successfully booked",
            "appointment details",
        ]
    )


def _ensure_full_doctor_list_in_reply(reply: str, doctor_list: str, next_step: str) -> str:
    if next_step != "doctor_id" or not doctor_list:
        return reply

    missing_doctors = [
        line
        for line in doctor_list.splitlines()
        if line.strip() and line.strip() not in reply
    ]
    if not missing_doctors:
        return reply

    # Keep doctor selection deterministic; HF may summarize away some rows.
    return reply.rstrip() + "\n\nAvailable doctors:\n" + doctor_list


def _generate_hf_reply(
    user_input: str,
    rule_reply: str,
    session: Dict[str, Any],
    doctor_list: str,
    next_step: str,
) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    if not HF_TOKEN or HF_TOKEN in {"PASTE_YOUR_HF_TOKEN_HERE", "hf_your_token_here"}:
        return None, "fallback", "HF token is missing.", None

    errors = []
    messages = _build_llm_messages(user_input, rule_reply, session, doctor_list, next_step)
    for model_name in DEFAULT_MODELS:
        try:
            hf_response = requests.post(
                HF_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 180,
                    "temperature": 0.35,
                },
                timeout=30,
            )
            if not hf_response.ok:
                errors.append(f"{model_name} -> {hf_response.status_code}: {hf_response.text[:160]}")
                continue

            data = hf_response.json()
            cleaned = data["choices"][0]["message"]["content"].strip()
            return (cleaned or None), "huggingface", None if cleaned else "Empty HF reply.", model_name
        except Exception as exc:
            errors.append(f"{model_name} -> {exc}")

    return None, "fallback", "HF failed for models: " + " | ".join(errors), None


def _finalize_chat_response(
    *,
    user_input: str,
    rule_reply: str,
    session_id: str,
    session: Dict[str, Any],
    doctor_list: str,
    next_step: str,
    booked: bool = False,
    doctor: Optional[Doctor] = None,
    appointment_id: Optional[int] = None,
) -> Dict[str, Any]:
    hf_reply, response_source, response_error, model_used = _generate_hf_reply(
        user_input=user_input,
        rule_reply=rule_reply,
        session=session,
        doctor_list=doctor_list,
        next_step=next_step,
    )
    reply = hf_reply or rule_reply
    if next_step == "doctor_id":
        # Doctor choice must come from the DB list, not from HF preference.
        reply = rule_reply
        response_source = "backend_guardrail"
        response_error = None
        model_used = None
    if not booked and _reply_claims_booking(reply):
        # State wins over phrasing: do not allow HF to confirm unsaved bookings.
        reply = rule_reply
        response_source = "backend_guardrail"
        response_error = "HF reply claimed a booking before the backend saved one."
        model_used = None
    reply = _ensure_full_doctor_list_in_reply(reply, doctor_list, next_step)
    reply = _ensure_booking_details_in_reply(reply, session, doctor, appointment_id, booked)
    session["history"].append({"role": "assistant", "content": reply})

    # Preserve session booking state so the user can continue (e.g., pay) after confirmation.
    # Do not reset the session automatically on booking — allow the frontend and user to confirm next steps.
    # If desired, a manual 'reset' action can clear the session.

    return {
        "session_id": session_id,
        "reply": reply,
        "booking_completed": booked,
        "appointment_id": appointment_id,
        "next_step": next_step,
        "response_source": response_source,
        "model_used": model_used,
        "response_error": response_error,
    }


def get_chat_reply(user_input: str, db: Session, session_id: Optional[str] = None):
    session_id, session = _get_session(session_id)
    doctor_list = _list_doctors(db)
    session.setdefault("history", [])
    session["history"].append({"role": "user", "content": user_input.strip()})

    name = _extract_name(user_input)
    if not name and not session["patient_name"]:
        stripped = user_input.strip()
        has_doctor_keyword = re.search(r"\b(?:doctor|dr\.?|doc)\b", stripped, re.IGNORECASE)
        if not has_doctor_keyword and _looks_like_plain_name(stripped):
            name = stripped.title()
    if name:
        session["patient_name"] = name

    wants_doctors = any(keyword in user_input.lower() for keyword in ["doctor", "available", "list"])
    wants_doctor_list_only = _is_doctor_listing_request(user_input)

    doctor = _extract_doctor(db, user_input)
    if not doctor and not session.get("doctor_id"):
        doctor = _extract_doctor_id_only(db, user_input)
    requested_doctor_name = _extract_requested_doctor_name(user_input)
    if doctor:
        session["doctor_id"] = str(doctor.id)
    elif session.get("doctor_id"):
        doctor = db.query(Doctor).filter(Doctor.id == int(session["doctor_id"])).first()

    date_time = _extract_date_time(user_input)
    if date_time:
        session["date_time"] = date_time
        session["pending_date_time"] = None
        session["pending_doctor_id"] = None
        session["pending_confirmation"] = None

    if not session["patient_name"] and not wants_doctor_list_only:
        if requested_doctor_name and not doctor:
            rule_reply = (
                "Please share the patient's name before I book the appointment.\n"
                f"I could not find a doctor named {requested_doctor_name}. "
                f"Please choose one of the available doctors:\n{doctor_list}"
            )
        else:
            rule_reply = "Please share the patient's name before I book the appointment."
        next_step = "patient_name"
        return _finalize_chat_response(
            user_input=user_input,
            rule_reply=rule_reply,
            session_id=session_id,
            session=session,
            doctor_list=doctor_list,
            next_step=next_step,
        )

    if session.get("pending_confirmation"):
        if _is_confirmation(user_input):
            if session.get("pending_doctor_id"):
                session["doctor_id"] = session["pending_doctor_id"]
                doctor = db.query(Doctor).filter(Doctor.id == int(session["doctor_id"])).first()
            if session.get("pending_date_time"):
                session["date_time"] = session["pending_date_time"]
            session["pending_doctor_id"] = None
            session["pending_date_time"] = None
            session["pending_confirmation"] = None
        elif _is_rejection(user_input):
            session["pending_doctor_id"] = None
            session["pending_date_time"] = None
            session["pending_confirmation"] = None
            rule_reply = "No problem. Please share another preferred date or time."
            next_step = "date_time" if session.get("doctor_id") else _get_next_step(session, False)
            return _finalize_chat_response(
                user_input=user_input,
                rule_reply=rule_reply,
                session_id=session_id,
                session=session,
                doctor_list=doctor_list,
                next_step=next_step,
            )

    relative_preference = _extract_relative_preference(user_input)
    if relative_preference and doctor and not date_time:
        session["pending_doctor_id"] = str(doctor.id)
        session["pending_date_time"] = _build_suggested_slot(relative_preference, doctor.available_time)
        session["pending_confirmation"] = relative_preference
        rule_reply = _build_availability_reply(doctor, relative_preference, session)
        next_step = "review"
        return _finalize_chat_response(
            user_input=user_input,
            rule_reply=rule_reply,
            session_id=session_id,
            session=session,
            doctor_list=doctor_list,
            next_step=next_step,
        )

    if requested_doctor_name and not doctor:
        rule_reply = (
            f"Sorry, I could not find a doctor named {requested_doctor_name}. "
            f"Please choose one of the available doctors:\n{doctor_list}"
        )
        next_step = "doctor_id"
        return _finalize_chat_response(
            user_input=user_input,
            rule_reply=rule_reply,
            session_id=session_id,
            session=session,
            doctor_list=doctor_list,
            next_step=next_step,
        )

    if doctor and session.get("date_time") and not _is_within_doctor_hours(doctor, session["date_time"]):
        session["date_time"] = None
        rule_reply = _build_out_of_hours_reply(doctor)
        next_step = "date_time"
        return _finalize_chat_response(
            user_input=user_input,
            rule_reply=rule_reply,
            session_id=session_id,
            session=session,
            doctor_list=doctor_list,
            next_step=next_step,
            doctor=doctor,
        )

    if wants_doctors and not session.get("doctor_id"):
        rule_reply = "Here are the available doctors:\n" + doctor_list
        next_step = _get_next_step(session, False)
        return _finalize_chat_response(
            user_input=user_input,
            rule_reply=rule_reply,
            session_id=session_id,
            session=session,
            doctor_list=doctor_list,
            next_step=next_step,
        )

    booked = False
    appointment_id = None
    booking_error = None
    # Only create an appointment if all required fields are present and an appointment hasn't
    # already been created for this session. This prevents duplicate bookings when the user
    # selects doctor or date/time in separate messages.
    if session["patient_name"] and session["doctor_id"] and session["date_time"] and not session.get("appointment_id"):
        try:
            appointment = Appointment(
                patient_name=session["patient_name"],
                doctor_id=int(session["doctor_id"]),
                date_time=session["date_time"],
            )
            db.add(appointment)
            db.commit()
            db.refresh(appointment)

            booked = True
            if appointment.id is None:
                raise ValueError("Appointment was saved without an id.")
            appointment_id = int(appointment.id)
            session["appointment_id"] = str(appointment_id)
            doctor = db.query(Doctor).filter(Doctor.id == int(session["doctor_id"])).first()
        except (SQLAlchemyError, ValueError) as exc:
            db.rollback()
            booking_error = f"Could not save the appointment: {exc.__class__.__name__}."
    elif session.get("appointment_id"):
        # If appointment already exists for this session, surface it (avoid creating another).
        booked = True
        appointment_id = int(session.get("appointment_id"))
        doctor = db.query(Doctor).filter(Doctor.id == int(session.get("doctor_id"))).first() if session.get("doctor_id") else None

    rule_reply = (
        booking_error
        or _build_rule_reply({**session, "doctor_list": doctor_list}, doctor, booked)
    )
    next_step = _get_next_step(session, booked)
    return _finalize_chat_response(
        user_input=user_input,
        rule_reply=rule_reply,
        session_id=session_id,
        session=session,
        doctor_list=doctor_list,
        next_step=next_step,
        booked=booked,
        doctor=doctor,
        appointment_id=appointment_id,
    )
