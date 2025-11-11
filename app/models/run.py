from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from db.database import Base

class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String)
    timestamp = Column(DateTime)

    agent_id = Column(Integer, ForeignKey("agents.id"))
    agent = relationship("Agent", back_populates="runs")