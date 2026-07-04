from dataclasses import dataclass


@dataclass
class Config:
    degenerate_face_eps: float = 1e-9
    min_wall_thickness: float = 1.0


DEFAULT_CONFIG = Config()
