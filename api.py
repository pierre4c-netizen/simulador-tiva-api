from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union
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
    isobolograma_2d: str = "Ninguno"
    tipo_grafico_iso: str = "Clásico"

app = FastAPI(title="TIVA Flow API Motor Matemático")

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

    @staticmethod
    def ketamina_pd(c: Union[float, np.ndarray], efecto: str) -> Union[float, np.ndarray]:
        if 'Analgesia Nociception Index' in efecto:
            return FarmacoMatematica.hill(c, 188.0, 12.5, 31.4, 100.0)
        elif 'Frecuencia Mediana EEG' in efecto:
            return FarmacoMatematica.hill(c, 2000.0, 3.8, 8.7, 1.1)
        elif 'Presión Arterial Sistólica' in efecto:
            return FarmacoMatematica.hill(c, 468.0, 2.04, 97.1, 148.7)
        elif 'Frecuencia Cardíaca' in efecto:
            return FarmacoMatematica.hill(c, 7580.0, 1.0, 72.7, 220.0)
        elif 'Intensidad Disociativa' in efecto:
            return FarmacoMatematica.hill(c, 242.48, 5.33, 0.0, 100.0)
        elif 'Factor Tolerancia' in efecto:
            return 1.0 + (c / 242.48)**1.31
        return np.full_like(c, np.nan) if isinstance(c, np.ndarray) else np.nan

    @staticmethod
    def herd_2008_prob_despertar(c_e: Union[float, np.ndarray], as_percent: bool = True) -> Union[float, np.ndarray]:
        ce_mg_l = np.clip(c_e, 0.0, None) / 1000.0
        logit_p0 = np.log(0.0247 / (1.0 - 0.0247))
        logit_p = np.clip(logit_p0 + 7.19 * ce_mg_l, -50.0, 50.0)
        p_despertar = 1.0 / (1.0 + np.exp(logit_p))
        return p_despertar * 100.0 if as_percent else p_despertar

    @staticmethod
    def herd_2008_sedacion(c_e: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ce_mg_l = np.clip(c_e, 0.0, None) / 1000.0
        c_pow = ce_mg_l ** 3.71
        ec50_pow = 0.56 ** 3.71
        num = 0.94 * c_pow
        den = ec50_pow + c_pow
        frac = np.where(den == 0, 0.0, num / den)
        return 5.0 * (1.0 - frac)

    @staticmethod
    def jonkman_2018_prob(ce_remi: Union[float, np.ndarray], ce_ket: Union[float, np.ndarray], as_percent: bool = True) -> Union[float, np.ndarray]:
        ce_remi_safe = np.clip(ce_remi, 0.0, None) if isinstance(ce_remi, np.ndarray) else max(0.0, ce_remi)
        ce_ket_safe = np.clip(ce_ket, 0.0, None) if isinstance(ce_ket, np.ndarray) else max(0.0, ce_ket)
        c50_1, g1 = 1.24, 8.44
        c50_2_basal, g2 = 0.46, 4.37
        cd2k, q = 254.0, 1.0  
        y1 = 1.0 / (1.0 + (ce_remi_safe / c50_1)**g1)
        c50_2_dyn = c50_2_basal * (1.0 + (ce_ket_safe / cd2k)**q)
        y2 = 1.0 / (1.0 + (ce_remi_safe / c50_2_dyn)**g2)
        p = 0.61 * y1 + 0.39 * y2
        return p * 100.0 if as_percent else p

    @staticmethod
    def jonkman_2018_iso_cx(ce_known: Union[float, np.ndarray], target_prob: float, solve_for: str) -> Union[float, np.ndarray]:
        if target_prob <= 0 or target_prob >= 1: return np.full_like(ce_known, np.nan) if isinstance(ce_known, np.ndarray) else np.nan
        
        def objective(cx, c_known, tgt, s_for):
            if s_for == 'ket': return float(FarmacoMatematica.jonkman_2018_prob(c_known, cx, False) - tgt)
            else: return float(FarmacoMatematica.jonkman_2018_prob(cx, c_known, False) - tgt)
                
        if isinstance(ce_known, np.ndarray):
            res = []
            for val in ce_known:
                try:
                    if solve_for == 'ket':
                        if objective(0.0, val, target_prob, 'ket') > 0: res.append(np.nan)
                        elif objective(50000.0, val, target_prob, 'ket') < 0: res.append(np.nan) 
                        else: res.append(root_scalar(lambda cx: objective(cx, val, target_prob, 'ket'), bracket=[0.0, 50000.0]).root)
                    else: 
                        if objective(0.0, val, target_prob, 'remi') < 0: res.append(np.nan)
                        else: res.append(root_scalar(lambda cx: objective(cx, val, target_prob, 'remi'), bracket=[0.0, 50.0]).root)
                except: res.append(np.nan)
            return np.array(res)
        else:
            try:
                if solve_for == 'ket':
                    if objective(0.0, ce_known, target_prob, 'ket') > 0 or objective(50000.0, ce_known, target_prob, 'ket') < 0: return np.nan
                    return root_scalar(lambda cx: objective(cx, ce_known, target_prob, 'ket'), bracket=[0.0, 50000.0]).root
                else: 
                    if objective(0.0, ce_known, target_prob, 'remi') < 0: return np.nan
                    return root_scalar(lambda cx: objective(cx, ce_known, target_prob, 'remi'), bracket=[0.0, 50.0]).root
            except: return np.nan

    @staticmethod
    def hui_1995_prob(c_prop: Union[float, np.ndarray], c_ket: Union[float, np.ndarray], efecto: str = 'LOC', as_percent: bool = True) -> Union[float, np.ndarray]:
        if 'Eléctrico' in efecto: c50_prop, c50_ket, gamma = 4.0, 1000.0, 5.33
        else: c50_prop, c50_ket, gamma = 2.4, 600.0, 4.75
            
        c_prop_safe = np.clip(c_prop, 0.0, None) if isinstance(c_prop, np.ndarray) else max(0.0, c_prop)
        c_ket_safe = np.clip(c_ket, 0.0, None) if isinstance(c_ket, np.ndarray) else max(0.0, c_ket)
        
        u = (c_prop_safe / c50_prop) + (c_ket_safe / c50_ket)
        p = (u**gamma) / (1.0 + u**gamma)
        return p * 100.0 if as_percent else p

    @staticmethod
    def hui_1995_iso_cx(c_known: Union[float, np.ndarray], target_prob: float, efecto: str = 'LOC', solve_for: str = 'prop') -> Union[float, np.ndarray]:
        if target_prob <= 0 or target_prob >= 1: return np.full_like(c_known, np.nan) if isinstance(c_known, np.ndarray) else np.nan
        
        if 'Eléctrico' in efecto: c50_prop, c50_ket, gamma = 4.0, 1000.0, 5.33
        else: c50_prop, c50_ket, gamma = 2.4, 600.0, 4.75
            
        u_req = (target_prob / (1.0 - target_prob)) ** (1.0 / gamma)
        c_known_safe = np.clip(c_known, 0.0, None) if isinstance(c_known, np.ndarray) else max(0.0, c_known)
        
        res = c50_prop * (u_req - (c_known_safe / c50_ket)) if solve_for == 'prop' else c50_ket * (u_req - (c_known_safe / c50_prop))
            
        if isinstance(res, np.ndarray): res[res < 0] = np.nan
        else: 
            if res < 0: return np.nan
        return res

    @staticmethod
    def hamp_2018_iso_csevo(c_ket: Union[float, np.ndarray], target_prob: float) -> Union[float, np.ndarray]:
        if abs(target_prob - 0.50) > 1e-5: return np.full_like(c_ket, np.nan) if isinstance(c_ket, np.ndarray) else np.nan
        c_ket_safe = np.clip(c_ket, 0.0, None) if isinstance(c_ket, np.ndarray) else max(0.0, c_ket)
        res = 2.2 - (2.28 * (c_ket_safe / (758.0 + c_ket_safe)))
        if isinstance(res, np.ndarray): res[res < 0] = np.nan
        else: 
            if res < 0: return np.nan
        return res

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

    @staticmethod
    def kuizenga_2019_c50(farmaco, ce_remi):
        if farmaco == 'Propofol': c50_base, t1, t2 = 1.63, -0.135, 0.816
        elif farmaco == 'Sevoflurano': c50_base, t1, t2 = 1.22, 0.0, 0.0
        else: return np.full_like(ce_remi, np.nan) if isinstance(ce_remi, np.ndarray) else np.nan

        is_array = isinstance(ce_remi, np.ndarray)
        ce_r = np.clip(ce_remi, 0.0, 4.0) if is_array else max(0.0, min(4.0, ce_remi))
        
        if is_array:
            INT = np.zeros_like(ce_r)
            m1 = ce_r <= 2.0
            m2 = ce_r > 2.0
            INT[m1] = (ce_r[m1] / 2.0) * t1
            INT[m2] = t1 + ((ce_r[m2] - 2.0) / 2.0) * (t1 * t2)
            return c50_base * (1.0 + INT)
        else:
            if ce_r <= 2.0: INT = (ce_r / 2.0) * t1
            else: INT = t1 + ((ce_r - 2.0) / 2.0) * (t1 * t2)
            return c50_base * (1.0 + INT)

    @staticmethod
    def kuizenga_2019_prob(c_hyp, farmaco, efecto, ce_remi):
        c50_adj = FarmacoMatematica.kuizenga_2019_c50(farmaco, ce_remi)
        if farmaco == 'Propofol': gamma, e0, emax = 3.41, 85.1, 63.1
        elif farmaco == 'Sevoflurano': gamma, e0, emax = 3.23, 85.1, 67.2
        else: return np.nan

        c_safe = np.clip(c_hyp, 0.0, None) if isinstance(c_hyp, np.ndarray) else max(0.0, c_hyp)
        return e0 - emax * ( (c_safe**gamma) / (c50_adj**gamma + c_safe**gamma) )

    @staticmethod
    def kuizenga_2019_iso_c(farmaco, efecto, ce_remi, target_psi):
        c50_adj = FarmacoMatematica.kuizenga_2019_c50(farmaco, ce_remi)
        if farmaco == 'Propofol': gamma, e0, emax = 3.41, 85.1, 63.1
        elif farmaco == 'Sevoflurano': gamma, e0, emax = 3.23, 85.1, 67.2
        else: return np.full_like(ce_remi, np.nan) if isinstance(ce_remi, np.ndarray) else np.nan

        if target_psi >= e0 or target_psi <= (e0 - emax):
            return np.full_like(ce_remi, np.nan) if isinstance(ce_remi, np.ndarray) else np.nan
            
        e_ratio = (e0 - target_psi) / emax
        k = (e_ratio / (1.0 - e_ratio)) ** (1.0 / gamma)
        return c50_adj * k

    @staticmethod
    def obtener_coordenadas_isobolicas(iso_v: str, ce_y: np.ndarray, edad: float, farmaco: str) -> dict:
        """Centraliza la matemática para sacar arrays de Req50, Req95, Base50 y Base95"""
        res = {'x_base': np.full_like(ce_y, np.nan), 'x95': np.full_like(ce_y, np.nan), 'x50': np.full_like(ce_y, np.nan), 
               'base_95': np.nan, 'base_50': np.nan, 'unit': ''}
        
        FM = FarmacoMatematica
        if farmaco == 'Fentanilo':
            if 'Smith 1994' in iso_v:
                b_50 = max(1.0, 4.9 - 0.09*(edad - 20)) if 'LOC' in iso_v else 15.2
                b_95 = b_50*(5.4/3.3) if 'LOC' in iso_v else 27.4
                p_val, g_val, c50_f = (0.50, 1.0, 0.75) if 'LOC' in iso_v else (0.95, 1.4, 0.63)
                fr_y = (p_val*ce_y**g_val)/(c50_f**g_val+ce_y**g_val)
                res.update({'x95': b_95*(1-fr_y), 'x50': b_50*(1-fr_y), 'base_95': b_95, 'base_50': b_50, 'unit': 'ug/ml'})
            elif 'Kazama 1998' in iso_v:
                ef_map = {'↓ FC 15%': 'Disminución FC 15%', '↓ FC 30%': 'Disminución FC 30%', '↓ PAS 15%': 'Disminución PAS 15%', '↓ PAS 30%': 'Disminución PAS 30%', '↓ PAS 40%': 'Disminución PAS 40%', 'No Movimiento': 'Respuesta Somática', 'No ↑ PAS 15%': 'Supresión Aumento PAS 15%'}
                ef_str = next((v for k, v in ef_map.items() if k in iso_v), 'Respuesta Somática')
                res.update({'x95': FM.kazama_1998_iso_cprop(ce_y, 0.95, ef_str), 'x50': FM.kazama_1998_iso_cprop(ce_y, 0.50, ef_str), 'base_95': FM.kazama_1998_iso_cprop(0.0, 0.95, ef_str), 'base_50': FM.kazama_1998_iso_cprop(0.0, 0.50, ef_str), 'unit': 'ug/ml'})
            elif 'Katoh' in iso_v:
                ef_str = 'LOC 1998' if 'LOC' in iso_v else ('Somática 1998' if '1998' in iso_v else ('Somática 1999' if 'Movimiento' in iso_v else 'Simpática 1999'))
                b50 = 0.62 if 'LOC' in iso_v else (1.84 if 'Somática 1998' == ef_str else (1.85 if 'Somática 1999' == ef_str else 4.15))
                b95 = 0.71 if 'LOC' in iso_v else (2.21 if 'Somática 1998' == ef_str else (2.28 if 'Somática 1999' == ef_str else 6.26))
                res.update({'x95': FM.katoh_iso_csevo(ce_y, 0.95, ef_str), 'x50': FM.katoh_iso_csevo(ce_y, 0.50, ef_str), 'base_95': b95, 'base_50': b50, 'unit': '%'})
            elif 'Vereecke 2013' in iso_v:
                is_mov = 'Movimiento' in iso_v; c50_f, c50_s, gamma, gamma_o = (2.07, 1.73, 6.40, 0.931) if is_mov else (0.43, 4.60, 6.40, 0.931)
                res.update({'x95': FM.vereecke_iso_cs(ce_y, c50_s, c50_f, gamma_o, gamma, 0.95), 'x50': FM.vereecke_iso_cs(ce_y, c50_s, c50_f, gamma_o, gamma, 0.50), 'base_95': (19.0**(1.0/gamma))*c50_s, 'base_50': 1.0*c50_s, 'unit': '%'})
            elif 'Wang 2026' in iso_v:
                is_prop = 'Propofol' in iso_v; c50_m = 9.77 if is_prop else 4.29
                res.update({'x95': FM.wang_2026_iso_cm(ce_y, c50_m, 0.95), 'x50': FM.wang_2026_iso_cm(ce_y, c50_m, 0.50), 'base_95': c50_m, 'base_50': c50_m, 'unit': 'ug/ml' if is_prop else '%'})

        elif farmaco == 'Remifentanilo':
            if 'Kuizenga 2019' in iso_v:
                is_prop = 'Propofol' in iso_v
                f_str = 'Propofol' if is_prop else 'Sevoflurano'
                res.update({'x95': FM.kuizenga_2019_iso_c(f_str, 'PSI', ce_y, 25.0), 'x50': FM.kuizenga_2019_iso_c(f_str, 'PSI', ce_y, 50.0), 'base_95': FM.kuizenga_2019_iso_c(f_str, 'PSI', 0.0, 25.0), 'base_50': FM.kuizenga_2019_iso_c(f_str, 'PSI', 0.0, 50.0), 'unit': 'ug/ml' if is_prop else '%'})
            elif 'Manyam 2006' in iso_v:
                pm = {'≤ 1':(7.30,7.84,0.23,3.94), '≥ 4':(4.19,4.25,0.28,0.58), 'mecanico':(3.82,2.43,0.54,1.27), 'termico':(3.38,1.32,0.55,3.47), 'electrico':(3.27,0.97,0.088,1.09), 'Laringoscopia':(3.70,2.36,0.54,1.22)}
                b0, b1, b2, b3 = next((v for k,v in pm.items() if k in iso_v), pm['≤ 1'])
                res.update({'x95': FM.manyam_iso_cs(ce_y,b0,b1,b2,b3,0.95), 'x50': FM.manyam_iso_cs(ce_y,b0,b1,b2,b3,0.50), 'base_95': (b0+2.9444)/b1, 'base_50': b0/b1, 'unit': '%'})
            elif 'Manyam 2007' in iso_v and 'BIS' in iso_v:
                ec_s, ec_r, alp, n = 2.37, 38.02, 0.52, 1.12
                res.update({'x95': FM.greco_iso_cs(ce_y,ec_s,ec_r,alp,n,0.4,invert=True), 'x50': FM.greco_iso_cs(ce_y,ec_s,ec_r,alp,n,0.6,invert=True), 'base_95': ec_s*((100.0/40 - 1.0)**(1.0/n)), 'base_50': ec_s*((100.0/60 - 1.0)**(1.0/n)), 'unit': '%'})
            elif 'Heyse 2014' in iso_v:
                if 'BIS' in iso_v: e0, gamma, c50_r, c50_s = 89.5, 1.88, 27.3, (2.29 if 'post' in iso_v else 1.99)
                elif 'SE ' in iso_v: e0, gamma, c50_r, c50_s = 97.1, 1.87, 16.2, (2.13 if 'post' in iso_v else 1.82)
                else: e0, gamma, c50_r, c50_s = 103.0, 2.08, 18.2, (2.24 if 'post' in iso_v else 1.88)
                res.update({'x95': FM.heyse_2014_iso_csevo(ce_y,40.0,e0,c50_s,c50_r,gamma), 'x50': FM.heyse_2014_iso_csevo(ce_y,60.0,e0,c50_s,c50_r,gamma), 'base_95': FM.heyse_2014_iso_csevo(0.0,40.0,e0,c50_s,c50_r,gamma), 'base_50': FM.heyse_2014_iso_csevo(0.0,60.0,e0,c50_s,c50_r,gamma), 'unit': '%'})
            elif 'Bouillon 2004' in iso_v:
                if 'BIS' in iso_v:
                    res.update({'x95': FM.bouillon_2004_bis_iso_cprop(ce_y,40.0), 'x50': FM.bouillon_2004_bis_iso_cprop(ce_y,60.0), 'base_95': FM.bouillon_2004_bis_iso_cprop(0.0,40.0), 'base_50': FM.bouillon_2004_bis_iso_cprop(0.0,60.0), 'unit': 'ug/ml'})
                else:
                    ef_str = 'LOC' if 'LOC' in iso_v else 'Laringoscopia'
                    res.update({'x95': FM.bouillon_2004_iso_cprop(ce_y,0.95,ef_str), 'x50': FM.bouillon_2004_iso_cprop(ce_y,0.50,ef_str), 'base_95': FM.bouillon_2004_iso_cprop(0.0,0.95,ef_str), 'base_50': FM.bouillon_2004_iso_cprop(0.0,0.50,ef_str), 'unit': 'ug/ml'})
            elif 'Jonkman 2018' in iso_v:
                res.update({'x95': FM.jonkman_2018_iso_cx(ce_y,0.50,'ket'), 'x50': FM.jonkman_2018_iso_cx(ce_y,0.80,'ket'), 'base_95': FM.jonkman_2018_iso_cx(0.0,0.50,'ket'), 'base_50': FM.jonkman_2018_iso_cx(0.0,0.80,'ket'), 'unit': 'ng/ml'})
            elif 'Kern 2004' in iso_v or 'Johnson' in iso_v:
                is_prop = 'Propofol' in iso_v; is_waking = '≥ 2' in iso_v or '≥ 4' in iso_v
                if is_prop:
                    if '≤ 3' in iso_v or '≥ 4' in iso_v: ec_s, ec_r, alp, n = 1.8, 12.5, 5.1, 5.8
                    elif 'Laringoscopia' in iso_v: ec_s, ec_r, alp, n = 5.60, 48.9, 33.2, 2.2
                    elif 'Mecanico' in iso_v: ec_s, ec_r, alp, n = 4.16, 8.84, 8.2, 8.3
                    elif 'Electrico' in iso_v: ec_s, ec_r, alp, n = 4.56, 21.3, 14.7, 6.0
                    elif '≤ 1' in iso_v: ec_s, ec_r, alp, n = 2.2, 33.1, 3.6, 5.0
                    elif '≥ 2' in iso_v: ec_s, ec_r, alp, n = 1.3, 10.5, 2.8, 3.5
                else:
                    if '≤ 1' in iso_v or '≥ 2' in iso_v or 'LOC' in iso_v: ec_s, ec_r, alp, n = 0.74, 50.9, 9.4, 5.2
                    elif 'mecanico' in iso_v.lower(): ec_s, ec_r, alp, n = 0.83, 1.3, 0.9, 2.7
                K50, K95 = (50.0/50.0)**(1.0/n), ((5.0/95.0)**(1.0/n) if is_waking else (95.0/5.0)**(1.0/n))
                res.update({'x95': FM.greco_iso_cs(ce_y,ec_s,ec_r,alp,n,0.95,is_waking), 'x50': FM.greco_iso_cs(ce_y,ec_s,ec_r,alp,n,0.50,is_waking), 'base_95': ec_s*K95, 'base_50': ec_s*K50, 'unit': 'ug/ml' if is_prop else '%'})

        elif farmaco == 'Ketamina':
            if 'Jonkman 2018' in iso_v:
                res.update({'x95': FM.jonkman_2018_iso_cx(ce_y,0.50,'remi'), 'x50': FM.jonkman_2018_iso_cx(ce_y,0.80,'remi'), 'base_95': FM.jonkman_2018_iso_cx(0.0,0.50,'remi'), 'base_50': FM.jonkman_2018_iso_cx(0.0,0.80,'remi'), 'unit': 'ng/ml'})
            elif 'Hui 1995' in iso_v:
                ef_str = 'Tolerancia al Estímulo Eléctrico' if 'Eléctrico' in iso_v else 'LOC'
                res.update({'x95': FM.hui_1995_iso_cx(ce_y,0.95,ef_str,'prop'), 'x50': FM.hui_1995_iso_cx(ce_y,0.50,ef_str,'prop'), 'base_95': FM.hui_1995_iso_cx(0.0,0.95,ef_str,'prop'), 'base_50': FM.hui_1995_iso_cx(0.0,0.50,ef_str,'prop'), 'unit': 'ug/ml'})
            elif 'Hamp 2018' in iso_v:
                res.update({'x95': np.full_like(ce_y, np.nan), 'x50': FM.hamp_2018_iso_csevo(ce_y, 0.50), 'base_95': np.nan, 'base_50': 2.2, 'unit': '%'})

        return res

# ==========================================
# 3. FARMACOCINÉTICA (PK) Y PROTECCIÓN
# ==========================================
def calcular_ke0_para_tpeak(k10: float, k12: float, k21: float, k13: float, k31: float, target_tpeak: float, default_ke0: float = 0.147) -> float:
    if k13 == 0.0 and k31 == 0.0:
        a = k10 + k12 + k21
        b = k10 * k21
        disc = max(0.0, a**2 - 4*b)
        l1 = (a + np.sqrt(disc)) / 2.0
        l2 = (a - np.sqrt(disc)) / 2.0
        
        if l1 == l2: return default_ke0
            
        A = (l1 - k21) / (l1 - l2)
        B = (k21 - l2) / (l1 - l2)
        
        def objective_2c(ke: float) -> float:
            def term(l, k, t): return t * np.exp(-k*t) if abs(k - l) < 1e-6 else (np.exp(-l*t) - np.exp(-k*t)) / (k - l)
            Cp_t = A * np.exp(-l1 * target_tpeak) + B * np.exp(-l2 * target_tpeak)
            Ce_t = ke * (A * term(l1, ke, target_tpeak) + B * term(l2, ke, target_tpeak))
            return Cp_t - Ce_t

        try: return root_scalar(objective_2c, bracket=[0.001, 100.0], method='brentq').root
        except: return default_ke0

    a = k10 + k12 + k13 + k21 + k31
    b = k10*(k21 + k31) + k12*k31 + k13*k21 + k21*k31
    c = k10 * k21 * k31
    roots = np.roots([1, a, b, c])
    lambdas = sorted(-np.real(roots[np.isreal(roots)]), reverse=True)
    
    if len(lambdas) < 3: return default_ke0
        
    l1, l2, l3 = lambdas[:3]
    A = (k21 - l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1))
    B = (k21 - l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2))
    C_coeff = (k21 - l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3))
    
    def get_term(l: float, ke: float, t: float) -> float: return t * np.exp(-ke * t) if abs(ke - l) < 1e-6 else (np.exp(-l * t) - np.exp(-ke * t)) / (ke - l)
    def objective_3c(ke: float) -> float: return (A * np.exp(-l1 * target_tpeak) + B * np.exp(-l2 * target_tpeak) + C_coeff * np.exp(-l3 * target_tpeak)) - ke * (A * get_term(l1, ke, target_tpeak) + B * get_term(l2, ke, target_tpeak) + C_coeff * get_term(l3, ke, target_tpeak))
        
    try: return root_scalar(objective_3c, bracket=[0.001, 100.0], method='brentq').root
    except: return default_ke0

def calcular_ffm(peso, altura, sexo, edad):
    altura_m = max(1.0, altura) / 100.0
    bmi = peso / (altura_m ** 2)
    wbm = (9270.0 * peso) / (6680.0 + 216.0 * bmi) if sexo == 'Masculino' else (9270.0 * peso) / (8780.0 + 244.0 * bmi)
    a_mat, A50, c_mat = (0.88, 13.4, 12.7) if sexo == 'Masculino' else (1.11, 7.1, 1.1)
    mf = a_mat + ((1.0 - a_mat) * (edad / A50)**c_mat) / (1.0 + (edad / A50)**c_mat) if edad > 0 else 1.0
    return mf * wbm

def _minto(p, a, s, e):
    lbm = (1.1*p - 128.0*((p/max(1.0, a))**2)) if s == 'Masculino' else (1.07*p - 148.0*((p/max(1.0, a))**2))
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

def _calcular_kamp_general(p, a, s, e):
    V1, V2, V3 = 21.0 * (p/70.0), 46.0 * (p/70.0), 254.0 * (p/70.0)
    Cl1, Cl2, Cl3 = (79.0/60.0) * ((p/70.0)**0.75), (97.0/60.0) * ((p/70.0)**0.75), (60.0/60.0) * ((p/70.0)**0.75)
    return V1, V2, V3, Cl1, Cl2, Cl3

def _calcular_herd_pediatrico(p, a, s, e):
    V1, V2, V3 = 38.7 * (p/70.0), 102.0 * (p/70.0), 0.0
    Cl1, Cl2, Cl3 = (90.0/60.0) * ((p/70.0)**0.75), (215.0/60.0) * ((p/70.0)**0.75), 0.0
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
    'Kamp 2020 (Propósito General / Alométrico)': _calcular_kamp_general,
    'Abuhelwa 2022 (Alométrico)': lambda p, a, s, e: (79.3*(p/70.0), 87.4*(p/70.0), 0.0, 1.16*((p/70.0)**0.75), (121.0/60.0)*((p/70.0)**0.75), 0.0),
    'Herd 2007 (Pediátrico / Alométrico)': _calcular_herd_pediatrico
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

    if 'Pediátrico' in modelo_pk and farmaco != 'Ketamina':
        ke0 = 0.0
    elif farmaco == 'Ketamina':
        if '0.238' in ke0_tpeak_str: ke0 = 0.238
        elif '1.83' in ke0_tpeak_str: ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 1.83, default_ke0=0.238)
        elif '11 seg' in ke0_tpeak_str: ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 11.0 / 60.0, default_ke0=3.78)
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
        elif '4 min' in ke0_tpeak_str: ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 4.0, default_ke0=0.147)
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
