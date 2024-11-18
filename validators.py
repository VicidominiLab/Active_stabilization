"""
Validation classes guaranteeing the launch of the program only with acceptable values and types of parameters

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

import os
from exceptions import BaseParticleException, FilenameError, ThrtypeError, ThrError, RError, RoiError, \
    PosAngleError, DzError, ZError, MidRndError, EpsilonError, MinptsError, DcError, DfrError, NfrMinError

class BaseValidator:

    @staticmethod
    def validate():
        raise NotImplemented


class FilenameValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, str):
            raise FilenameError(f"filename must be string got {type(value)}") 
        
        if not os.path.exists(value):
            raise FilenameError(f"file {value} does not exist.") 


class ThrtypeValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, str):
            raise ThrtypeError(f"thrtype must be string got {type(value)}")
        
        if value not in ('topfraction', 'topvalue'):
            raise ThrtypeError(f"thrtype must be either 'topfraction' nor 'topvalue'")


class ThrValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, float)) and (not isinstance(value, int)):
            raise ThrError(f"thr must be float or int. Got {str(type(value))}")
        
        if value <= 0.5:
            raise ThrError(f"thr must be more that 0.5, got {value}") 


class RValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, float)) and (not isinstance(value, int)):
            raise RError(f"R must be float or int. Got {str(type(value))}")

        if value <= 0:
            raise RError(f"R must be positive, got {value}")


class RoiValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, list)) and (not isinstance(value, tuple)):
            raise RoiError(f"roi must be list or tuple of length 4, got {str(type(value))}")
        if (len(value) != 4):
            raise RoiError(f"roi must be list or tuple of length 4, got {str(type(value))}")


class PosAngleValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, int)):
            raise PosAngleError(f"positiveAngle must be int, got {str(type(value))}")

        if value < 0:
            raise PosAngleError(f"positiveAngle must be positive, got {value}")


class DzValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, float):
            raise DzError(f"dz must be float, got {str(type(value))}")

        if value <= 0:
            raise DzError(f"dz must be positive, got {value}")


class ZValidator(BaseValidator):
    #int,
    @staticmethod
    def validate(value):
        if (not isinstance(value, int)):
            raise ZError(f"z must be int, got {str(type(value))}")


class MidRndValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, int)):
            raise MidRndError(f"mid_rng must be int, got {str(type(value))}")

        if (value > 180) or (value < -180):
            raise MidRndError(f"mid_rng must be in range [-180..180]")


class EpsilonValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if (not isinstance(value, float)) and (not isinstance(value, int)):
            raise EpsilonError(f"epsilon must be float or int. Got {str(type(value))}")

        if value <= 0:
            raise EpsilonError(f"epsilon must be positive, got {value}")


class MinptsValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, int):
            raise MinptsError(f"minpts must be int. Got {str(type(value))}")

        if value <= 0:
            raise MinptsError(f"minpts must be positive, got {value}")


class DcValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, int):
            raise DcError(f"dc must be int. Got {str(type(value))}")

        if value <= 0:
            raise DcError(f"dc must be positive, got {value}")


class DfrValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, int):
            raise DfrError(f"dfr must be int. Got {str(type(value))}")

        if value <= 0:
            raise DfrError(f"dfr must be positive, got {value}")


class NfrMinValidator(BaseValidator):

    @staticmethod
    def validate(value):
        if not isinstance(value, int):
            raise NfrMinError(f"Nfr_min must be int. Got {str(type(value))}")

        if value <= 0:
            raise NfrMinError(f"Nfr_min must be positive, got {value}")


validators = {
    "filename": FilenameValidator,
    "thrtype": ThrtypeValidator,
    "thr": ThrValidator,
    "R": RValidator,
    "roi": RoiValidator,
    "positiveAngle": PosAngleValidator,
    "dz": DzValidator,
    "z0": ZValidator,
    "mid_rng": MidRndValidator,
    "epsilon": EpsilonValidator,
    "minpts": MinptsValidator,
    "dc": DcValidator,
    "dfr": DfrValidator,
    "Nfr_min": NfrMinValidator


}

def validate(f):
    def wrapper(*args, **kwargs):
        errors = []
        for key, value in kwargs.items():
            if key in validators:
                validator = validators[key]
                try:
                    validator.validate(value)
                except BaseParticleException as exc:
                    errors.append(str(exc))
        if errors:
            raise SystemExit("Some input data is incorrect: \n - " +"\n\r - ".join(errors))
        return f(*args, **kwargs)
    return wrapper
