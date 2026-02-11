from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from db.db import Base

class ResearchDB(Base):
    __tablename__ = 'research_data'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(default='research_example')
    type_of_research: Mapped[str]
    picture: Mapped[bytes] 

    patient_id: Mapped[int] = mapped_column(ForeignKey('patient_data.id'), nullable=False)
    patient: Mapped[int] = relationship('PatientDB', back_populates='research')
    
    company_id: Mapped[int] = mapped_column(ForeignKey('company_data.id'), nullable=False)
    company: Mapped[str] = relationship('CompanyDB', back_populates='research')
    