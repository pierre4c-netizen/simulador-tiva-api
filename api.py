from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Union
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

# ==========================================
# 1. ESTRUCTURAS DE ENTRADA
# ==========================================
class EventoTIVA(BaseModel):
    tipo: str
    ini_min: float
    fin_min: float
    tasa_ug_min: float

class PeticionSimulacion(BaseModel):
    farmaco: str = "Fentanilo"
    peso_kg: float
    altura_cm: float
    edad_anos: float
    sexo: str
    modelo_pk: str
    ke0_tpeak: str
    eventos: List[EventoTIVA]
    minutos_simulacion: int = 1440
    modelo_3d: str = "Ninguna"

app = FastAPI(title="TIVA Flow API Motor Matemático")

# Habilitar CORS para evitar problemas si consultas desde Flutter Web/App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CLASE MATEMÁTICA PURA (PD e Isobolas)
# ==========================================
class FarmacoMatematica:
    @staticmethod
    def hill(c, c50, gamma, e0=0.0, emax=100.0):
        return e0 + (emax - e0) * (c**gamma) / (c50**gamma + c**gamma)

    @staticmethod
    def greco_prob(cs, cr, ec_s, ec_r, alpha, n, invert=False, as_percent=True):
        u = (cs/ec_s) + (cr/ec_r) + alpha * (cs/ec_s) * (cr/ec_r)
        p = (u**n) / (u**n + 1.0)
        if invert: p = 1.0 - p
        return p * 100.0 if as_percent else p

    @staticmethod
    def greco_iso_cs(cr, ec_s, ec_r, alpha, n, target_prob, invert=False):
        p = 1.0 - target_prob if invert else target_prob
        if p <= 0 or p >= 1: return np.full_like(cr, np.nan) if isinstance(cr, np.ndarray) else np.nan
        k = (p / (1.0 - p))**(1.0/n)
        den = 1.0 + alpha * (cr/ec_r)
        cs = ec_s * (k - cr/ec_r) / den
        if isinstance(cs, np.ndarray):
            cs[cs < 0] = np.nan
            cs[den == 0] = np.nan
        else:
            if cs < 0 or den == 0: return np.nan
        return cs

    @staticmethod
    def manyam_prob(cs, cr, b0, b1, b2, b3, as_percent=True):
        x = b0 - b1*cs - b2*cr - b3*cs*cr
        p = 1.0 / (1.0 + np.exp(x))
        return p * 100.0 if as_percent else p

    @staticmethod
    def manyam_iso_cs(cr, b0, b1, b2, b3, target_prob):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(cr, np.nan) if isinstance(cr, np.ndarray) else np.nan
        logit_term = np.log((1.0 - target_prob) / target_prob)
        den = b1 + b3*cr
        cs = (b0 - b2*cr - logit_term) / den
        if isinstance(cs, np.ndarray):
            cs[cs < 0] = np.nan
            cs[den == 0] = np.nan
        else:
            if cs < 0 or den == 0: return np.nan
        return cs

    @staticmethod
    def vereecke_prob(cs, cr, c50_s, c50_f, gamma_o, gamma, as_percent=True):
        u = (cs/c50_s) * (1.0 + (cr/c50_f)**gamma_o)
        p = (u**gamma) / (1.0 + u**gamma)
        return p * 100.0 if as_percent else p

    @staticmethod
    def vereecke_iso_cs(cr, c50_s, c50_f, gamma_o, gamma, target_prob):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(cr, np.nan) if isinstance(cr, np.ndarray) else np.nan
        u_req = (target_prob / (1.0 - target_prob))**(1.0/gamma)
        return u_req * c50_s / (1.0 + (cr/c50_f)**gamma_o)

    @staticmethod
    def wang_2026_prob(cm, cf, c50_m, as_percent=True):
        um = cm/c50_m
        uf = cf/2.58
        u = um + uf
        u_safe = np.where(u == 0, 1e-6, u)
        x = um/u_safe
        y = uf/u_safe
        logU50 = (1.0-x)*(1.0-y)*(-0.06*x - 1.35*y - 0.78*x*y)
        U50 = 10.0**logU50
        n = 1.98*x + 1.98*y + 1.44*x*y
        p = ((u/U50)**n) / (1.0 + (u/U50)**n)
        p = np.where(u == 0, 0, p)
        return p * 100.0 if as_percent else p

    @staticmethod
    def wang_2026_iso_cm(cf, c50_m, target_prob):
        if isinstance(cf, np.ndarray):
            res = []
            for val in cf:
                try:
                    if (FarmacoMatematica.wang_2026_prob(30.0, val, c50_m, False) - target_prob) > 0:
                        res.append(root_scalar(lambda cm: FarmacoMatematica.wang_2026_prob(cm, val, c50_m, False) - target_prob, bracket=[0, 30]).root)
                    else: res.append(np.nan)
                except: res.append(np.nan)
            return np.array(res)
        return np.nan

    @staticmethod
    def bouillon_2004_prob(c_prop, c_remi, efecto='Laringoscopia', as_percent=True):
        c50_remi, c50_prop = 1.07, 8.04
        gamma_remi, gamma_prop = 0.97, 5.1
        i_pre = 0.60 if efecto == 'LOC' else 1.05 
        c_prop_safe = np.clip(c_prop, 1e-6, None) if isinstance(c_prop, np.ndarray) else max(1e-6, c_prop)
        c_remi_safe = np.clip(c_remi, 1e-6, None) if isinstance(c_remi, np.ndarray) else max(1e-6, c_remi)
        num_remi = c_remi_safe ** gamma_remi
        den_remi = num_remi + (c50_remi * i_pre) ** gamma_remi
        i_post = i_pre * (1.0 - (num_remi / den_remi))
        c50_efectiva = c50_prop * i_post
        num_prop = c_prop_safe ** gamma_prop
        den_prop = num_prop + (c50_efectiva ** gamma_prop)
        p_no_respuesta = num_prop / den_prop
        return p_no_respuesta * 100.0 if as_percent else p_no_respuesta

    @staticmethod
    def bouillon_2004_iso_cprop(c_remi, target_prob, efecto='Laringoscopia'):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(c_remi, np.nan) if isinstance(c_remi, np.ndarray) else np.nan
        c50_remi, c50_prop = 1.07, 8.04
        gamma_remi, gamma_prop = 0.97, 5.1
        i_pre = 0.60 if efecto == 'LOC' else 1.05 
        c_remi_safe = np.clip(c_remi, 1e-6, None) if isinstance(c_remi, np.ndarray) else max(1e-6, c_remi)
        num_remi = c_remi_safe ** gamma_remi
        den_remi = num_remi + (c50_remi * i_pre) ** gamma_remi
        i_post = i_pre * (1.0 - (num_remi / den_remi))
        term = (target_prob / (1.0 - target_prob)) ** (1.0 / gamma_prop)
        return (c50_prop * i_post) * term
        
    @staticmethod
    def bouillon_2004_bis(c_prop, c_remi):
        c_prop_safe = np.clip(c_prop, 0.0, None) if isinstance(c_prop, np.ndarray) else max(0.0, c_prop)
        c_remi_safe = np.clip(c_remi, 0.0, None) if isinstance(c_remi, np.ndarray) else max(0.0, c_remi)
        u = (c_prop_safe / 4.47) + (c_remi_safe / 19.3)
        return 97.4 - 97.4 * ((u ** 1.43) / (1.0 + (u ** 1.43)))

    @staticmethod
    def bouillon_2004_bis_iso_cprop(c_remi, target_bis):
        if target_bis <= 0 or target_bis >= 97.4: return np.full_like(c_remi, np.nan) if isinstance(c_remi, np.ndarray) else np.nan
        effect_ratio = (97.4 - target_bis) / 97.4
        u_req = (effect_ratio / (1.0 - effect_ratio)) ** (1.0 / 1.43)
        c_remi_safe = np.clip(c_remi, 0.0, None) if isinstance(c_remi, np.ndarray) else max(0.0, c_remi)
        c_prop = 4.47 * (u_req - (c_remi_safe / 19.3))
        if isinstance(c_prop, np.ndarray): c_prop[c_prop < 0] = np.nan
        else: 
            if c_prop < 0: return np.nan
        return c_prop

    @staticmethod
    def kazama_1998_prob(c_prop, c_fent, efecto, as_percent=True):
        params = {
            'Disminución PAS 15%': (3.6, 9.7, 1.5, 1.5), 'Disminución PAS 30%': (8.1, 20.5, 3.1, 1.6),
            'Disminución PAS 40%': (17.7, 195.1, 41.5, 8.5), 'Disminución FC 15%': (14.4, 3.5, 1.65, 3.3),
            'Disminución FC 30%': (20.5, 6.7, 1.2, 4.6), 'Respuesta Somática': (13.8, 9.7, 6.8, 2.63),
            'Supresión Aumento PAS 15%': (27.7, 5.3, 3.7, 1.7)
        }
        if efecto not in params: return np.full_like(c_prop, np.nan) if isinstance(c_prop, np.ndarray) else np.nan
        ec_s, ec_r, alpha, n = params[efecto]
        return FarmacoMatematica.greco_prob(c_prop, c_fent, ec_s, ec_r, alpha, n, invert=False, as_percent=as_percent)

    @staticmethod
    def kazama_1998_iso_cprop(c_fent, target_prob, efecto):
        params = {
            'Disminución PAS 15%': (3.6, 9.7, 1.5, 1.5), 'Disminución PAS 30%': (8.1, 20.5, 3.1, 1.6),
            'Disminución PAS 40%': (17.7, 195.1, 41.5, 8.5), 'Disminución FC 15%': (14.4, 3.5, 1.65, 3.3),
            'Disminución FC 30%': (20.5, 6.7, 1.2, 4.6), 'Respuesta Somática': (13.8, 9.7, 6.8, 2.63),
            'Supresión Aumento PAS 15%': (27.7, 5.3, 3.7, 1.7)
        }
        if efecto not in params or target_prob <= 0 or target_prob >= 1: return np.full_like(c_fent, np.nan) if isinstance(c_fent, np.ndarray) else np.nan
        ec_s, ec_r, alpha, n = params[efecto]
        return FarmacoMatematica.greco_iso_cs(c_fent, ec_s, ec_r, alpha, n, target_prob, invert=False)

    @staticmethod
    def katoh_prob(c_sevo, c_fent, efecto, as_percent=True):
        if efecto == 'LOC 1998': base_50, base_95, c50_fent, gamma_fent, max_red = 0.62, 0.71, 7.3, 1.2, 1.0
        elif efecto == 'Somática 1998': base_50, base_95, c50_fent, gamma_fent, max_red = 1.84, 2.21, 1.8, 1.0, 0.80
        elif efecto == 'Somática 1999': base_50, base_95, c50_fent, gamma_fent, max_red = 1.85, 2.28, 1.8, 1.0, 0.80
        elif efecto == 'Simpática 1999': base_50, base_95, c50_fent, gamma_fent, max_red = 4.15, 6.26, 0.78, 1.5, 0.95
        else: return np.full_like(c_sevo, np.nan) if isinstance(c_sevo, np.ndarray) else np.nan

        fr = (max_red * c_fent**gamma_fent) / (c50_fent**gamma_fent + c_fent**gamma_fent)
        u50 = np.clip(base_50 * (1.0 - fr), 1e-5, None) if isinstance(base_50 * (1.0 - fr), np.ndarray) else max(1e-5, base_50 * (1.0 - fr))
        ratio = np.clip((base_95 * (1.0 - fr)) / u50, 1.01, None) if isinstance((base_95 * (1.0 - fr)) / u50, np.ndarray) else max(1.01, (base_95 * (1.0 - fr)) / u50)
        gamma = np.log(19.0) / np.log(ratio)
        p = (c_sevo**gamma) / (c_sevo**gamma + u50**gamma)
        return p * 100.0 if as_percent else p

    @staticmethod
    def katoh_iso_csevo(c_fent, target_prob, efecto):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(c_fent, np.nan) if isinstance(c_fent, np.ndarray) else np.nan
        if efecto == 'LOC 1998': base_50, base_95, c50_fent, gamma_fent, max_red = 0.62, 0.71, 7.3, 1.2, 1.0
        elif efecto == 'Somática 1998': base_50, base_95, c50_fent, gamma_fent, max_red = 1.84, 2.21, 1.8, 1.0, 0.80
        elif efecto == 'Somática 1999': base_50, base_95, c50_fent, gamma_fent, max_red = 1.85, 2.28, 1.8, 1.0, 0.80
        elif efecto == 'Simpática 1999': base_50, base_95, c50_fent, gamma_fent, max_red = 4.15, 6.26, 0.78, 1.5, 0.95
        else: return np.full_like(c_fent, np.nan) if isinstance(c_fent, np.ndarray) else np.nan

        fr = (max_red * c_fent**gamma_fent) / (c50_fent**gamma_fent + c_fent**gamma_fent)
        u50 = np.clip(base_50 * (1.0 - fr), 1e-5, None) if isinstance(base_50 * (1.0 - fr), np.ndarray) else max(1e-5, base_50 * (1.0 - fr))
        ratio = np.clip((base_95 * (1.0 - fr)) / u50, 1.01, None) if isinstance((base_95 * (1.0 - fr)) / u50, np.ndarray) else max(1.01, (base_95 * (1.0 - fr)) / u50)
        gamma = np.log(19.0) / np.log(ratio)
        return u50 * ((target_prob / (1.0 - target_prob))**(1.0 / gamma))

    # --- NUEVOS MODELOS KETAMINA (PD) ---
    @staticmethod
    def ketamina_pd(c: Union[float, np.ndarray], efecto: str) -> Union[float, np.ndarray]:
        if 'Analgesia Nociception Index' in efecto:
            return FarmacoMatematica.hill(c, 188.0, 12.5, 31.4, 100.0)
        elif 'Presión Arterial Sistólica' in efecto:
            return FarmacoMatematica.hill(c, 468.0, 2.04, 97.1, 148.7)
        elif 'Frecuencia Cardíaca' in efecto:
            return FarmacoMatematica.hill(c, 7580.0, 1.0, 72.7, 220.0)
        elif 'Intensidad Disociativa' in efecto:
            return FarmacoMatematica.hill(c, 242.48, 5.33, 0.0, 100.0)
        elif 'Factor Tolerancia' in efecto or 'Factor de Tolerancia' in efecto:
            return 1.0 + (c / 242.48)**1.31
        return np.full_like(c, np.nan) if isinstance(c, np.ndarray) else np.nan

    # --- NUEVOS MODELOS EEG HEYSE 2014 ---
    @staticmethod
    def heyse_2014_prob(c_sevo, c_remi, e0, c50_sevo, c50_remi, gamma):
        u = (c_sevo / c50_sevo) + (c_remi / c50_remi)
        return e0 - e0 * ((u**gamma) / (1.0 + u**gamma))

    @staticmethod
    def heyse_2014_iso_csevo(c_remi, target_effect, e0, c50_sevo, c50_remi, gamma):
        if target_effect <= 0 or target_effect >= e0: 
            return np.full_like(c_remi, np.nan) if isinstance(c_remi, np.ndarray) else np.nan
        u_req = ((e0 - target_effect) / target_effect) ** (1.0 / gamma)
        c_sevo = c50_sevo * (u_req - (c_remi / c50_remi))
        if isinstance(c_sevo, np.ndarray): 
            c_sevo[c_sevo < 0] = np.nan
        else:
            if c_sevo < 0: return np.nan
        return c_sevo

# ==========================================
# 3. FARMACOCINÉTICA (PK) Y PROTECCIÓN
# ==========================================
def calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, target_tpeak):
    a = k10 + k12 + k13 + k21 + k31
    b = k10*(k21+k31) + k12*k31 + k13*k21 + k21*k31
    c = k10*k21*k31
    roots = np.roots([1, a, b, c])
    lambdas = sorted(-np.real(roots[np.isreal(roots)]), reverse=True)
    if len(lambdas) < 3: return 0.147
    l1, l2, l3 = lambdas[:3]
    A = (k21 - l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1))
    B = (k21 - l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2))
    C_coeff = (k21 - l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3))
    def get_term(l, ke, t): return t * np.exp(-ke * t) if abs(ke - l) < 1e-6 else (np.exp(-l * t) - np.exp(-ke * t)) / (ke - l)
    def obj(ke): return (A * np.exp(-l1 * target_tpeak) + B * np.exp(-l2 * target_tpeak) + C_coeff * np.exp(-l3 * target_tpeak)) - ke * (A * get_term(l1, ke, target_tpeak) + B * get_term(l2, ke, target_tpeak) + C_coeff * get_term(l3, ke, target_tpeak))
    try: return root_scalar(obj, bracket=[0.001, 3.0], method='brentq').root
    except: return 0.147

def calcular_ffm(peso, altura, sexo, edad):
    bmi = peso / ((max(1.0, altura) / 100.0) ** 2)
    wbm = (9270.0 * peso) / (6680.0 + 216.0 * bmi) if sexo == 'Masculino' else (9270.0 * peso) / (8780.0 + 244.0 * bmi)
    a_mat, A50, c_mat = (0.88, 13.4, 12.7) if sexo == 'Masculino' else (1.11, 7.1, 1.1)
    mf = a_mat + ((1.0 - a_mat) * (edad / A50)**c_mat) / (1.0 + (edad / A50)**c_mat) if edad > 0 else 1.0
    return mf * wbm

def _minto(p, a, s, e):
    lbm = (1.1*p - 128.0*((p/a)**2)) if s == 'Masculino' else (1.07*p - 148.0*((p/a)**2))
    V1 = max(0.01, 5.1-0.0201*(e-40)+0.072*(lbm-55))
    V2 = max(0.01, 9.82-0.0811*(e-40)+0.108*(lbm-55))
    Cl1 = max(0.001, 2.6-0.0162*(e-40)+0.0191*(lbm-55))
    Cl2 = max(0.001, 2.05-0.0301*(e-40))
    Cl3 = max(0.001, 0.076-0.00113*(e-40))
    return V1, V2, 5.42, Cl1, Cl2, Cl3

def _lacolla(p, a, s, e):
    ffm = calcular_ffm(p, a, s, e)
    V1 = max(0.01, 5.1-0.0201*(e-40)+0.072*(ffm-55))
    V2 = max(0.01, 9.82-0.0811*(e-40)+0.108*(ffm-55))
    Cl1 = max(0.001, 2.6-0.0162*(e-40)+0.0191*(ffm-55))
    Cl2 = max(0.001, 2.05-0.0301*(e-40))
    Cl3 = max(0.001, 0.076-0.00113*(e-40))
    return V1, V2, 5.42, Cl1, Cl2, Cl3

def _kim(p, a, s, e):
    ffm = calcular_ffm(p, a, s, e)
    V1 = max(0.01, 4.76*((p/74.5)**0.658))
    V2 = max(0.01, 8.4*((ffm/52.3)**0.573)-0.0936*(e-37))
    V3 = max(0.01, 4.0-0.0477*(e-37))
    Cl1 = max(0.001, 2.77*((p/74.5)**0.336)-0.0149*(e-37))
    Cl2 = max(0.001, 1.94-0.0280*(e-37))
    return V1, V2, V3, Cl1, Cl2, 0.197

def _eleveld(p, a, s, e):
    SIZE = calcular_ffm(p, a, s, e) / calcular_ffm(70.0, 170.0, 'Masculino', 35.0)
    KMAT = ((p**2.0)/(p**2.0 + 2.88**2.0)) / ((70.0**2.0)/(70.0**2.0 + 2.88**2.0))
    KSEX = 1.0 if s == 'Masculino' else 1.0 + 0.470 * ((e**6.0)/(e**6.0 + 12.0**6.0)) * (1.0 - ((e**6.0)/(e**6.0 + 45.0**6.0)))
    V1 = max(0.01, 5.81*SIZE*np.exp(-0.00554*(e-35.0)))
    V2 = max(0.01, 8.82*SIZE*np.exp(-0.00327*(e-35.0))*KSEX)
    V3 = max(0.01, 5.03*SIZE*np.exp(-0.0315*(e-35.0))*np.exp(-0.0260*(p-70.0)))
    Cl1 = max(0.001, 2.58*(SIZE**0.75)*KMAT*KSEX*np.exp(-0.00327*(e-35.0)))
    Cl2 = max(0.001, 1.72*(((max(0.01, 8.82*SIZE*np.exp(-0.00327*(e-35.0))*KSEX))/8.82)**0.75)*np.exp(-0.00554*(e-35.0))*KSEX)
    Cl3 = max(0.001, 0.124*(((max(0.01, 5.03*SIZE*np.exp(-0.0315*(e-35.0))*np.exp(-0.0260*(p-70.0))))/5.03)**0.75)*np.exp(-0.00554*(e-35.0)))
    return V1, V2, V3, Cl1, Cl2, Cl3

PK_DISPATCHER = {
    'Scott 1987 (Fijo)': lambda p, a, s, e: (12.7, 50.7, 274.0, 0.574, 4.01, 1.95),
    'Shafer 1990 (Fijo)': lambda p, a, s, e: (6.09, 28.1, 228.0, 0.504, 2.87, 1.37),
    'Shafer 1990 (Peso Corporal Total)': lambda p, a, s, e: (0.105*p, 0.446*p, 3.37*p, 0.00838*p, 0.0474*p, 0.0199*p),
    'Bae 2020 (Alométrico)': lambda p, a, s, e: (10.1*((p/70.0)**1.23), 26.5*((p/70.0)**1.23), 206.0*((p/70.0)**1.23), 0.704*((p/70.0)**0.313), 2.38*((p/70.0)**0.313), 1.49*((p/70.0)**0.313)),
    'Ginsberg 1996 (Pediátrico / Peso Corporal Total y Edad)': lambda p, a, s, e: (max(0.001, 0.43*(p-19.8)+5.8), max(0.001, 6.2*(e-6.4)+34.4), 0.0, max(0.001, 0.01*(p-19.8)+0.35), max(0.001, 0.82), 0.0),
    'Maharaj 2019 (Pediátrico / Alométrico)': lambda p, a, s, e: (10.8*(p/70.0), 417.0*(p/70.0), 0.0, (32.5/60.0)*((p/70.0)**0.75), (104.0/60.0)*((p/70.0)**0.75), 0.0),
    'Egan 1996 (Fijo)': lambda p, a, s, e: (7.6, 9.4, 4.7, 2.92, 1.95, 0.10),
    'Rigby-Jones 2007 (Pediátrico / Alométrico)': lambda p, a, s, e: (0.963*(p/10.5), 1.480*(p/10.5), 0.0, 0.716*((p/10.5)**0.75), 0.840*((p/10.5)**0.75), 0.0),
    'Staschen 2013 (Pediátrico / Alométrico dependiente de peso)': lambda p, a, s, e: (1.44*((p/14.6)**0.81), 3.02*((p/14.6)**0.74), 0.0, 1.09*((p/14.6)**(1.32*(p**-0.20))), 0.63*((p/14.6)**0.70), 0.0),
    'Minto 1997 (Masa corporal magra y Edad)': _minto,
    'La Colla 2009 (Masa libre de grasa y Edad)': _lacolla,
    'Kim-Obara-Egan 2017 (Alométrico y Edad)': _kim,
    'Eleveld 2017 (Propósito General / Alométrico, Edad y Sexo)': _eleveld,
    'Clements125 1981 (Peso Corporal Total)': lambda p, a, s, e: (1.202*p, 2.114*p, 0.0, 0.0166*p, 0.0263*p, 0.0),
    'Clements250 1981 (Peso Corporal Total)': lambda p, a, s, e: (1.70*p, 2.40*p, 0.0, 0.0191*p, 0.0317*p, 0.0),
    'Domino 1982 (Fijo)': lambda p, a, s, e: (3.47, 11.0, 134.0, 1.296, 2.276, 2.058),
    'Domino 1984 (Peso Corporal Total)': lambda p, a, s, e: (0.063*p, 0.1511*p, 2.5517*p, 0.0276*p, 0.0373*p, 0.0372*p),
    'Kamp 2020 (Alométrico)': lambda p, a, s, e: (25.8*(p/70.0), 115.0*(p/70.0), 0.0, 1.78*((p/70.0)**0.75), 2.1*((p/70.0)**0.75), 0.0),
    'Abuhelwa 2022 (Alométrico)': lambda p, a, s, e: (79.3*(p/70.0), 87.4*(p/70.0), 0.0, 1.16*((p/70.0)**0.75), (121.0/60.0)*((p/70.0)**0.75), 0.0)
}

def get_pk_params(farmaco, modelo_pk, ke0_tpeak_str, peso, altura, sexo, edad):
    if modelo_pk in PK_DISPATCHER:
        V1, V2, V3, Cl1, Cl2, Cl3 = PK_DISPATCHER[modelo_pk](peso, altura, sexo, edad)
    else:
        V1, V2, V3, Cl1, Cl2, Cl3 = 12.7, 50.7, 274.0, 0.574, 4.01, 1.95

    k10 = Cl1/V1 if V1 > 0 else 0.0
    k12 = Cl2/V1 if V1 > 0 else 0.0
    k21 = Cl2/V2 if V2 > 0 else 0.0
    k13 = Cl3/V1 if V1 > 0 else 0.0
    k31 = Cl3/V3 if V3 > 0 else 0.0

    if 'Pediátrico' in modelo_pk:
        ke0 = 0.0
    elif farmaco == 'Ketamina':
        if '0.238' in ke0_tpeak_str: ke0 = 0.238
        elif '1.83' in ke0_tpeak_str: ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 1.83)
        elif '0.0835' in ke0_tpeak_str: ke0 = 0.0835
        else: ke0 = 0.0
    elif farmaco == 'Remifentanilo':
        if 'Abad' in ke0_tpeak_str: ke0 = 0.120
        elif 'Egan' in ke0_tpeak_str: ke0 = 0.433
        elif 'Eleveld' in ke0_tpeak_str: ke0 = max(0.01, 1.09 * float(np.exp(-0.0289 * (edad - 35.0))))
        else: ke0 = max(0.01, 0.595 - 0.007 * (edad - 40))
    else:
        if '0.108' in ke0_tpeak_str: ke0 = 0.108
        elif '0.105' in ke0_tpeak_str: ke0 = 0.105
        elif '4 min' in ke0_tpeak_str: ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 4.0)
        else: ke0 = 0.147

    return k10, k12, k21, k13, k31, ke0, V1, V2, V3

def clean_arr(arr):
    res = []
    for x in arr:
        if x is None:
            res.append(None)
        else:
            val = np.real(x)
            if np.isnan(val) or np.isinf(val): res.append(None)
            else: res.append(float(val))
    return res

# ==========================================
# 4. ENDPOINT ÚNICO UNIFICADO
# ==========================================
@app.post("/simular")
def calcular_simulacion_completa(datos: PeticionSimulacion) -> Dict[str, Any]:
    k10, k12, k21, k13, k31, ke0, V1, V2, V3 = get_pk_params(
        datos.farmaco, datos.modelo_pk, datos.ke0_tpeak, datos.peso_kg, 
        datos.altura_cm, datos.sexo, datos.edad_anos
    )

    def ode_sys(y, t_ode):
        x1, x2, x3, ce, ce_rig, ce_abad = y
        entrada_total = sum([ev.tasa_ug_min for ev in datos.eventos if ev.ini_min <= t_ode <= ev.fin_min])
        dx1 = entrada_total - (k10 + k12 + k13)*x1 + k21*x2 + k31*x3
        return [dx1, k12*x1 - k21*x2, k13*x1 - k31*x3, ke0 * ((x1 / V1) - ce), 0.054 * ((x1 / V1) - ce_rig), 0.12 * ((x1 / V1) - ce_abad)]

    max_evento_min = max([e.fin_min for e in datos.eventos]) if datos.eventos else 60
    minutos_totales = max(datos.minutos_simulacion, int(max_evento_min) + 60)
    t_sim = np.linspace(0, float(minutos_totales), (minutos_totales * 10) + 1)
    
    sol = odeint(ode_sys, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], t_sim)
    Cp = np.clip(sol[:, 0] / V1, 0.0, None)
    Ce = np.clip(sol[:, 3], 0.0, None)
    Ce_abad = np.clip(sol[:, 5], 0.0, None)

    # === ALERTAS CLÍNICAS ===
    torax_lenoso, depresion_resp, apnea = np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool)
    if 'Pediátrico' not in datos.modelo_pk:
        if datos.farmaco == 'Fentanilo':
            grad_cp = np.gradient(Cp, t_sim)
            is_rig = False
            for i in range(len(t_sim)):
                if not is_rig and Cp[i] >= 21.5 and grad_cp[i] > 0.1: is_rig = True
                elif is_rig and Cp[i] <= 6.9: is_rig = False
                torax_lenoso[i] = is_rig
            depresion_resp = Ce >= 1.0
        elif datos.farmaco == 'Remifentanilo':
            apnea = Ce >= 1.5

    # === MODELOS FARMACODINÁMICOS (PD) ===
    pd_arrays = {}
    if 'Pediátrico' not in datos.modelo_pk:
        if datos.farmaco == 'Fentanilo':
            pd_arrays['Frecuencia Borde Espectral (Scott 1985)'] = FarmacoMatematica.hill(Ce, 6.9, 4.9, 19.2, 5.1).tolist()
            pd_arrays['Frecuencia Borde Espectral (Scott 1987)'] = FarmacoMatematica.hill(Ce, max(0.1, 11.4 - 0.0675 * datos.edad_anos), 4.3, 18.9, 5.9).tolist()
            pd_arrays['Frecuencia Borde Espectral (Scott 1991)'] = FarmacoMatematica.hill(Ce, 8.1, 6.2, 25.0, 8.2).tolist()
            pd_arrays['Probabilidad EVA < 5 al aplicar 20N (Bae 2020)'] = FarmacoMatematica.hill(Ce, 0.63, 2.24, 0.0, 100.0).tolist()
            pd_arrays['Poder Theta (Balanza 2022)'] = ((Ce - 5.5) / 0.55).tolist()
            pd_arrays['Indice Ventilacion Minuto (Balanza 2022)'] = np.clip(-1.62 * ((Ce - 5.5) / 0.55) + 83.8, 0, 100).tolist()
            pd_arrays['LOC (Balanza 2022)'] = (100.0 / (1.0 + np.exp(-(2.6 - 0.1508 * ((Ce - 5.5) / 0.55))))).tolist()
            pd_arrays['Ventilacion Minuto (Mildh 2001)'] = (9.9 * (1.0 - (Ce / (5.49 + Ce)))).tolist()
            pd_arrays['Frecuencia Respiratoria (Mildh 2001)'] = (15.1 * (1.0 - (Ce / (3.15 + Ce)))).tolist()
            pd_arrays['PaCO2 (Mildh 2001)'] = (40.503 + (3.915 * Ce)).tolist()
        elif datos.farmaco == 'Ketamina':
            pd_arrays['Analgesia Nociception Index (ANI) (Navarrete 2025)'] = FarmacoMatematica.ketamina_pd(Ce, 'Analgesia Nociception Index (ANI) (Navarrete 2025)').tolist()
            pd_arrays['Presión Arterial Sistólica (Abuhelwa 2022)'] = FarmacoMatematica.ketamina_pd(Cp, 'Presión Arterial Sistólica (Abuhelwa 2022)').tolist()
            pd_arrays['Frecuencia Cardíaca (Abuhelwa 2022)'] = FarmacoMatematica.ketamina_pd(Cp, 'Frecuencia Cardíaca (Abuhelwa 2022)').tolist()
            pd_arrays['Intensidad Disociativa (Olofsen 2022)'] = FarmacoMatematica.ketamina_pd(Ce, 'Intensidad Disociativa (Olofsen 2022)').tolist()
            pd_arrays['Factor Tolerancia al Estimulo Mecanico (Olofsen 2022)'] = FarmacoMatematica.ketamina_pd(Ce, 'Factor Tolerancia al Estimulo Mecanico (Olofsen 2022)').tolist()
        else: # Remifentanilo
            pd_arrays['Frecuencia Borde Espectral (Minto 1997)'] = FarmacoMatematica.hill(Ce, max(0.1, 13.1 - 0.148 * (datos.edad_anos - 40)), 2.44, 20.0, 5.5).tolist()
            pd_arrays['Frecuencia Borde Espectral (Egan 1996)'] = FarmacoMatematica.hill(Ce, 19.9, 4.3, 19.0, 5.2).tolist()
            pd_arrays['Frecuencia Borde Espectral (Eleveld 2017)'] = FarmacoMatematica.hill(Ce, 12.7, 2.87, 19.9, 5.66).tolist()
            pd_arrays['Factor Tolerancia al Estimulo Mecanico (Abad 2022)'] = FarmacoMatematica.hill(Ce_abad, 2.8, 1.9, 1.0, 3.88).tolist()

    # === INTERACCIONES PD (ISOBOLAS Y 3D) ===
    isobolas = {}
    superficie_3d = {}

    if 'Pediátrico' not in datos.modelo_pk:
        max_y = max(np.max(Cp), np.max(Ce)) * 1.1 if max(np.max(Cp), np.max(Ce)) > 0 else 10.0
        ce_y = np.linspace(0, max_y, 100)
        isobolas['ce_farmaco'] = ce_y.tolist()
        
        if datos.farmaco == 'Fentanilo':
            b_50 = max(1.0, 4.9 - 0.09*(datos.edad_anos - 20)); b_95 = b_50*(5.4/3.3); fr_sm = (0.50*ce_y)/(0.75+ce_y)
            isobolas['Propofol LOC (Smith 1994)'] = {'ce_50': clean_arr(b_50*(1-fr_sm)), 'ce_95': clean_arr(b_95*(1-fr_sm))}
            fr_som = (0.95*ce_y**1.4)/(0.63**1.4+ce_y**1.4)
            isobolas['Propofol No Movimiento (Smith 1994)'] = {'ce_50': clean_arr(15.2*(1-fr_som)), 'ce_95': clean_arr(27.4*(1-fr_som))}
            
            eff_map_kazama = {
                'Propofol ↓ FC 15% (Kazama 1998)': 'Disminución FC 15%', 'Propofol ↓ FC 30% (Kazama 1998)': 'Disminución FC 30%',
                'Propofol No Movimiento (Kazama 1998)': 'Respuesta Somática', 'Propofol No ↑ PAS 15% (Kazama 1998)': 'Supresión Aumento PAS 15%',
                'Propofol ↓ PAS 15% (Kazama 1998)': 'Disminución PAS 15%', 'Propofol ↓ PAS 30% (Kazama 1998)': 'Disminución PAS 30%',
                'Propofol ↓ PAS 40% (Kazama 1998)': 'Disminución PAS 40%'
            }
            for ui_name, eff in eff_map_kazama.items():
                isobolas[ui_name] = {'ce_50': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.50, eff)), 'ce_95': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.95, eff))}
            
            eff_map_katoh = {
                'Sevoflurano LOC (Katoh 1998)': 'LOC 1998', 'Sevoflurano No ↑ FC 15% / ↑ MAP 15% (Katoh 1999)': 'Simpática 1999',
                'Sevoflurano No Movimiento (Katoh 1998)': 'Somática 1998', 'Sevoflurano No Movimiento (Katoh 1999)': 'Somática 1999'
            }
            for ui_name, eff in eff_map_katoh.items():
                isobolas[ui_name] = {'ce_50': clean_arr(FarmacoMatematica.katoh_iso_csevo(ce_y, 0.50, eff)), 'ce_95': clean_arr(FarmacoMatematica.katoh_iso_csevo(ce_y, 0.95, eff))}
            
            isobolas['Sevoflurano No Movimiento (Vereecke 2013)'] = {'ce_50': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 1.73, 2.07, 0.931, 6.40, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 1.73, 2.07, 0.931, 6.40, 0.95))}
            isobolas['Sevoflurano No ↑ FC 15% / ↑ PAS 15% (Vereecke 2013)'] = {'ce_50': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 4.60, 0.43, 0.931, 6.40, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 4.60, 0.43, 0.931, 6.40, 0.95))}
            isobolas['Propofol Tolerancia LMA ΔANI < 20% (Wang 2026)'] = {'ce_50': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 9.77, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 9.77, 0.95))}
            isobolas['Sevoflurano Tolerancia LMA ΔANI < 20% (Wang 2026)'] = {'ce_50': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 4.29, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 4.29, 0.95))}
        
        elif datos.farmaco == 'Remifentanilo':
            isobolas['Propofol BIS (Bouillon 2004)'] = {'ce_50': clean_arr(FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_y, 60.0)), 'ce_95': clean_arr(FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_y, 40.0))}
            isobolas['Propofol LOC (Bouillon 2004)'] = {'ce_50': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.50, 'LOC')), 'ce_95': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.95, 'LOC'))}
            isobolas['Propofol Tolerancia Laringoscopia (Bouillon 2004)'] = {'ce_50': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.50, 'Laringoscopia')), 'ce_95': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.95, 'Laringoscopia'))}

            p_greco = {
                'Propofol MOAA/S ≤ 1 (Johnson 2008)': (2.2, 33.1, 3.6, 5.0, False), 'Propofol MOAA/S ≤ 3 (Kern 2004)': (1.8, 12.5, 5.1, 5.8, False),
                'Propofol MOAA/S ≥ 2 (Johnson 2008)': (1.3, 10.5, 2.8, 3.5, True), 'Propofol MOAA/S ≥ 4 (Kern 2004)': (1.8, 12.5, 5.1, 5.8, True),
                'Propofol Tolerancia al Estimulo Electrico (Kern 2004)': (4.56, 21.3, 14.7, 6.0, False), 'Propofol Tolerancia al Estimulo Mecanico (Kern 2004)': (4.16, 8.84, 8.2, 8.3, False),
                'Propofol Tolerancia Laringoscopia (Kern 2004)': (5.60, 48.9, 33.2, 2.2, False), 'Sevoflurano MOAA/S ≤ 1 (Johnson 2010)': (0.74, 50.9, 9.4, 5.2, False),
                'Sevoflurano MOAA/S ≥ 2 (Johnson 2010)': (0.74, 50.9, 9.4, 5.2, True), 'Sevoflurano Tolerancia al estimulo mecanico (Johnson 2010)': (0.83, 1.3, 0.9, 2.7, False)
            }
            for k, v in p_greco.items(): isobolas[k] = {'ce_50': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.50, v[4])), 'ce_95': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.95, v[4]))}
            isobolas["Sevoflurano BIS (Manyam 2007)"] = {'ce_50': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, 2.37, 38.02, 0.52, 1.12, 0.60, True)), 'ce_95': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, 2.37, 38.02, 0.52, 1.12, 0.40, True))}

            p_man = {
                'Sevoflurano MOAA/S ≤ 1 (Manyam 2006)': (7.30, 7.84, 0.23, 3.94), 'Sevoflurano MOAA/S ≥ 4 (Manyam 2006)': (4.19, 4.25, 0.28, 0.58),
                'Sevoflurano Tolerancia al estimulo mecanico (Manyam 2006)': (3.82, 2.43, 0.54, 1.27), 'Sevoflurano Tolerancia al estimulo termico (Manyam 2006)': (3.38, 1.32, 0.55, 3.47),
                'Sevoflurano Tolerancia al estimulo electrico (Manyam 2006)': (3.27, 0.97, 0.088, 1.09), 'Sevoflurano Tolerancia Laringoscopia (Manyam 2006)': (3.70, 2.36, 0.54, 1.22)
            }
            for k, v in p_man.items(): isobolas[k] = {'ce_50': clean_arr(FarmacoMatematica.manyam_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.50)), 'ce_95': clean_arr(FarmacoMatematica.manyam_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.95))}

            # Heyse 2014 EEG
            e0_bis, gam_bis, c50_r_bis, c50_s_bisp, c50_s_biso = 89.5, 1.88, 27.3, 2.29, 1.99
            isobolas['Sevoflurano BIS post-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis))}
            isobolas['Sevoflurano BIS pre-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis))}
            
            e0_se, gam_se, c50_r_se, c50_s_sep, c50_s_seo = 97.1, 1.87, 16.2, 2.13, 1.82
            isobolas['Sevoflurano SE post-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_se, c50_s_sep, c50_r_se, gam_se)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_se, c50_s_sep, c50_r_se, gam_se))}
            isobolas['Sevoflurano SE pre-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_se, c50_s_seo, c50_r_se, gam_se)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_se, c50_s_seo, c50_r_se, gam_se))}
            
            e0_re, gam_re, c50_r_re, c50_s_rep, c50_s_reo = 103.0, 2.08, 18.2, 2.24, 1.88
            isobolas['Sevoflurano RE post-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_re, c50_s_rep, c50_r_re, gam_re)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_re, c50_s_rep, c50_r_re, gam_re))}
            isobolas['Sevoflurano RE pre-Laringoscopia (Heyse 2014)'] = {'ce_50': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 60.0, e0_re, c50_s_reo, c50_r_re, gam_re)), 'ce_95': clean_arr(FarmacoMatematica.heyse_2014_iso_csevo(ce_y, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re))}

        # SUPERFICIE 3D UNIFICADA
        if datos.modelo_3d != 'Ninguna' and datos.modelo_3d in isobolas:
            is_prop = 'Propofol' in datos.modelo_3d
            if datos.farmaco == 'Fentanilo':
                y_max, x_max = 10.0, 30.0 if is_prop else 8.0
                ylab = 'Ce Fentanilo (ng/ml)'
            elif datos.farmaco == 'Remifentanilo':
                y_max, x_max = 15.0, 15.0 if is_prop else 8.0
                ylab = 'Ce Remifentanilo (ng/ml)'
            else:
                y_max, x_max = 10.0, 10.0
                ylab = 'Ce Ketamina (ng/ml)'
                
            xlab = 'Ce Propofol (ug/ml)' if is_prop else 'etSEV (%)'
            zlab = 'Valor BIS' if 'BIS' in datos.modelo_3d else 'Probabilidad (%)'
            
            res = 25
            x_m = np.linspace(0, x_max, res)
            y_m = np.linspace(0, y_max, res)
            X, Y = np.meshgrid(x_m, y_m)
            Z = np.zeros_like(X)
            
            if datos.farmaco == 'Fentanilo':
                if 'Smith 1994' in datos.modelo_3d:
                    if 'LOC' in datos.modelo_3d:
                        b_50 = max(1.0, 4.9 - 0.09*(datos.edad_anos - 20)); b_95 = b_50*(5.4/3.3); fr = (0.50*Y)/(0.75+Y)
                    else:
                        b_50 = 15.2; b_95 = 27.4; fr = (0.95*Y**1.4)/(0.63**1.4+Y**1.4)
                    u50 = np.clip(b_50 * (1.0 - fr), 1e-5, None); ratio = np.clip((b_95 * (1.0 - fr)) / u50, 1.01, None)
                    Z = (X**(np.log(19.0)/np.log(ratio))) / (X**(np.log(19.0)/np.log(ratio)) + u50**(np.log(19.0)/np.log(ratio))) * 100.0
                elif 'Kazama 1998' in datos.modelo_3d:
                    Z = FarmacoMatematica.kazama_1998_prob(X, Y, eff_map_kazama[datos.modelo_3d], True)
                elif 'Katoh' in datos.modelo_3d:
                    Z = FarmacoMatematica.katoh_prob(X, Y, eff_map_katoh[datos.modelo_3d], True)
                elif 'Vereecke 2013' in datos.modelo_3d:
                    is_mov = 'Movimiento' in datos.modelo_3d; c50_f = 2.07 if is_mov else 0.43; c50_s = 1.73 if is_mov else 4.60
                    Z = FarmacoMatematica.vereecke_prob(X, Y, c50_s, c50_f, 0.931, 6.40, True)
                elif 'Wang 2026' in datos.modelo_3d:
                    Z = FarmacoMatematica.wang_2026_prob(X, Y, 9.77 if is_prop else 4.29, True)
            elif datos.farmaco == 'Remifentanilo':
                if 'Bouillon 2004' in datos.modelo_3d:
                    if 'BIS' in datos.modelo_3d: Z = FarmacoMatematica.bouillon_2004_bis(X, Y)
                    else: Z = FarmacoMatematica.bouillon_2004_prob(X, Y, 'LOC' if 'LOC' in datos.modelo_3d else 'Laringoscopia', True)
                elif 'Heyse 2014' in datos.modelo_3d:
                    if 'BIS' in datos.modelo_3d:
                        e0, gamma, c50_remi = 89.5, 1.88, 27.3
                        c50_sevo = 2.29 if 'post' in datos.modelo_3d else 1.99
                    elif 'SE ' in datos.modelo_3d:
                        e0, gamma, c50_remi = 97.1, 1.87, 16.2
                        c50_sevo = 2.13 if 'post' in datos.modelo_3d else 1.82
                    elif 'RE ' in datos.modelo_3d:
                        e0, gamma, c50_remi = 103.0, 2.08, 18.2
                        c50_sevo = 2.24 if 'post' in datos.modelo_3d else 1.88
                    Z = FarmacoMatematica.heyse_2014_prob(X, Y, e0, c50_sevo, c50_remi, gamma)
                elif 'Johnson' in datos.modelo_3d or 'Kern' in datos.modelo_3d:
                    v = p_greco[datos.modelo_3d]
                    Z = FarmacoMatematica.greco_prob(X, Y, v[0], v[1], v[2], v[3], invert=v[4], as_percent=True)
                elif 'Manyam 2007' in datos.modelo_3d:
                    Z = FarmacoMatematica.greco_prob(X, Y, 2.37, 38.02, 0.52, 1.12, invert=True, as_percent=True)
                elif 'Manyam 2006' in datos.modelo_3d:
                    v = p_man[datos.modelo_3d]
                    Z = FarmacoMatematica.manyam_prob(X, Y, v[0], v[1], v[2], v[3], True)

            Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=0.0)
            superficie_3d = {
                "x_mesh": x_m.tolist(), "y_mesh": y_m.tolist(), "z_mesh": np.round(Z, 2).tolist(),
                "iso50_x": clean_arr(isobolas[datos.modelo_3d]['ce_50']), "iso50_y": clean_arr(isobolas['ce_farmaco']),
                "iso95_x": clean_arr(isobolas[datos.modelo_3d]['ce_95']), "iso95_y": clean_arr(isobolas['ce_farmaco']),
                "xlabel": xlab, "ylabel": ylab, "zlabel": zlab, "title": datos.modelo_3d
            }

    return {
        "tiempo_minutos": np.round(t_sim, 2).tolist(),
        "cp": np.round(Cp, 3).tolist(), "ce": np.round(Ce, 3).tolist(),
        "alertas_clinicas": {"torax_lenoso": torax_lenoso.tolist(), "depresion_respiratoria": depresion_resp.tolist(), "apnea": apnea.tolist()},
        "farmacodinamia_pd": {k: np.round(v, 2).tolist() for k, v in pd_arrays.items()},
        "isobolas_interaccion": isobolas,
        "superficie_3d": superficie_3d
    }
