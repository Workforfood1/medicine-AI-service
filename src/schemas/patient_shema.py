from pydantic import BaseModel

# Для пациента нужно вписать: пол, возраст, курит ли он, пьет и тд 
class Patient(BaseModel):
    pass
