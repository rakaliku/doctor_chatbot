import os

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
# from dotenv import load_dotenv

# load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hospital.db")

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
    doctor = relationship("Doctor")
    vaccine = relationship("Vaccine")


def init_db():
    Base.metadata.create_all(bind=engine)


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
