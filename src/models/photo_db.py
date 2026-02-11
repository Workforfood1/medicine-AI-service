from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from db.db import Base

class PhotoDB(Base):
    __tablename__ = 'photo_data'
    id: Mapped[int] = mapped_column(primary_key=True)
    photo: Mapped[bytes] # Изменить принимаемый тип данных
    potology: Mapped[str]
    
    research_id: Mapped[int] = mapped_column(ForeignKey('research_data.id'), nullable=False)
    research: Mapped[int] = relationship('ResearchDB', back_populates='picture')