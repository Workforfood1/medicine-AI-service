from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.db import Base

class CompanyDB(Base):
    __tablename__ = 'company_data'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(default='company_example')
    
    research: Mapped[str] = relationship('ResearchDB', back_populates='company')