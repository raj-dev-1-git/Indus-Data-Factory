from sqlalchemy import Column, Integer, String, Float

from app.database import Base


class AudioResult(Base):

    __tablename__ = "audio_results"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String)

    transcript = Column(String)

    reasoning = Column(String)
