from sqlalchemy import Column, Integer, String, Text
from database import Base

class Work(Base):
    __tablename__ = "works"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    image_url = Column(String)
    link = Column(String)  # Can be an external URL or an internal route like /tool/detection
    category = Column(String, default="Web")
