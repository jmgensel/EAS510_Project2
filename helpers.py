import math



def gaussian(value, mean, sd):
    try: return (
            (1 / (sd * math.sqrt(2 * math.pi))) *
            math.exp(-((value - mean) ** 2) / (2 * (sd ** 2))) if sd!=0 else 0
    )
    except: return 0

