import os
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
# from dotenv import load_dotenv

# load_dotenv()

# Use a project-absolute SQLite path so the DB is consistent regardless of process cwd
BASE_DIR = Path(__file__).resolve().parent
_default_db = BASE_DIR / "hospital.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_default_db}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    specialization = Column(String)
    available_time = Column(String)


class Vaccine(Base):
    __tablename__ = "vaccines"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    stock = Column(Integer)


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    vaccine_id = Column(Integer, ForeignKey("vaccines.id"), nullable=True)
    date_time = Column(String)
    # Payment fields (amount stored in paise)
    amount_due = Column(Integer, default=0)
    payment_status = Column(String, default="pending")  # pending, paid
    razorpay_order_id = Column(String, nullable=True)
    razorpay_payment_id = Column(String, nullable=True)
    doctor = relationship("Doctor")
    vaccine = relationship("Vaccine")


def init_db():
    Base.metadata.create_all(bind=engine)

    # SQLite-only best-effort migrations: add missing payment columns when upgrading an existing sqlite DB
    # For Postgres (Supabase), rely on SQLAlchemy migrations or manual ALTERs. Avoid running SQLite PRAGMA on Postgres.
    if DATABASE_URL.startswith("sqlite"):
        try:
            with engine.connect() as conn:
                res = conn.execute(text("PRAGMA table_info('appointments')"))
                cols = [row[1] for row in res.fetchall()]
                if "amount_due" not in cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN amount_due INTEGER DEFAULT 0"))
                if "payment_status" not in cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN payment_status VARCHAR DEFAULT 'pending'"))
                if "razorpay_order_id" not in cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN razorpay_order_id VARCHAR"))
                if "razorpay_payment_id" not in cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN razorpay_payment_id VARCHAR"))
        except Exception:
            # Best-effort migration; ignore failures here so startup still proceeds and errors surface at operation time
            pass
    else:
        # For Postgres (Supabase) we do not attempt PRAGMA-based migrations here.
        # In production, use a proper migration tool (Alembic) or run ALTER TABLE statements explicitly as needed.
        pass


def seed_sample_data():
    db = SessionLocal()
    try:
        if not db.query(Doctor).first():
            doctors = [
                Doctor(name="Dr. Ramesh Kumar", specialization="Pediatrics", available_time="10:00-16:00"),
                Doctor(name="Dr. Priya Singh", specialization="General Medicine", available_time="09:30-14:00"),
                Doctor(name="Dr. Arun Patel", specialization="Immunization Specialist", available_time="10:30-15:00"),
            ]
            vaccines = [
                Vaccine(name="MMR", stock=5),
                Vaccine(name="Polio", stock=10),
                Vaccine(name="DTaP", stock=2),
            ]
            db.add_all(doctors + vaccines)
            db.commit()
    finally:
        db.close()
