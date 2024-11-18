"""

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

class BaseParticleException(Exception):
    pass

class ParticleException(Exception):
    pass

class FilenameError(BaseParticleException):
    pass

class ThrtypeError(BaseParticleException):
    pass

class ThrError(BaseParticleException):
    pass

class RError(BaseParticleException):
    pass

class RoiError(BaseParticleException):
    pass

class PosAngleError(BaseParticleException):
    pass

class DzError(BaseParticleException):
    pass

class ZError(BaseParticleException):
    pass

class MidRndError(BaseParticleException):
    pass

class EpsilonError(BaseParticleException):
    pass

class MinptsError(BaseParticleException):
    pass

class DcError(BaseParticleException):
    pass

class DfrError(BaseParticleException):
    pass

class NfrMinError(BaseParticleException):
    pass