from sqlalchemy.orm import Mapped, mapped_column, relationship


from db.db import Base

class PatientDB(Base):
    __tablename__ = 'patient_data'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(default='patient')
    
    research: Mapped[int] = relationship('ResearchDB', back_populates='patient')
    