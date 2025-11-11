from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from db.database import Base


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    model = Column(String)

    project_id = Column(Integer, ForeignKey("projects.id"))
    project = relationship("Project", back_populates="agents")

    runs = relationship("Run", back_populates="agent")