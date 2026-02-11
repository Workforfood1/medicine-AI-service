from sqlalchemy.orm import Mapped, mapped_column

from db.db import Base

class PhotoData(Base):
    __tablename__ = 'photo_data'
    id: Mapped[int] = mapped_column(primary_key=True)
    