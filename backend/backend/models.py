from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    phone = Column(String)
    location = Column(String)
    bio = Column(Text)
    birth_date = Column(DateTime)
    gender = Column(String)
    height = Column(Float)
    weight = Column(Float)
    blood_type = Column(String)
    notifications_enabled = Column(Boolean, default=True)
    dark_mode = Column(Boolean, default=False)
    language = Column(String, default='en')
    
    user = relationship("User", back_populates="profile")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    age = Column(Float)
    sex = Column(String)
    ethnicity = Column(String)
    smoking_status = Column(String)
    bmi = Column(Float)
    
    hypertension = Column(Integer, default=0)
    diabetes = Column(Integer, default=0)
    asthma = Column(Integer, default=0)
    copd = Column(Integer, default=0)
    cancer = Column(Integer, default=0)
    obesity = Column(Integer, default=0)
    anxiety_or_depression = Column(Integer, default=0)
    psychiatric_disorder = Column(Integer, default=0)
    
    sleep_disorder_note_count = Column(Integer, default=0)
    insomnia_billing_code_count = Column(Integer, default=0)
    anx_depr_billing_code_count = Column(Integer, default=0)
    psych_note_count = Column(Integer, default=0)
    insomnia_rx_count = Column(Integer, default=0)
    joint_disorder_billing_code_count = Column(Integer, default=0)
    emr_fact_count = Column(Float, default=0)
    
    insomnia_probability = Column(Float)
    insomnia_class = Column(Integer)
    risk_level = Column(String)
    
    created_at = Column(DateTime, server_default=func.now())
    
    user = relationship("User", back_populates="predictions")
