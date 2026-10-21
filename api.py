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
    # Nuevos parámetros para la magia dinámica (Opcionales para no romper la app actual)
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
        elif 'Factor de Tolerancia' in efecto or 'Factor Tolerancia' in efecto:
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
            if s_for == 'ket':
                return float(FarmacoMatematica.jonkman_2018_prob(c_known, cx, False) - tgt)
            else:
                return float(FarmacoMatematica.jonkman_2018_prob(cx, c_known, False) - tgt)
                
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
    def estimar_bis_global(etsev: float, ce_remi: float) -> float: 
        return 100.0 / (((etsev/2.37) + (ce_remi/38.02) + 0.52*(etsev/2.37)*(ce_remi/38.02))**1.12 + 1.0)
        
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

    # ==============================================================
    # MOTOR DE EXTRACCIÓN DINÁMICA (Para la nueva Opción B)
    # ==============================================================
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
        x1, x2, x3, ce = y
        entrada_total = sum([ev.tasa_ug_min for ev in datos.eventos if ev.ini_min <= t_ode <= ev.fin_min])
        dx1 = entrada_total - (k10 + k12 + k13)*x1 + k21*x2 + k31*x3
        return [dx1, k12*x1 - k21*x2, k13*x1 - k31*x3, ke0 * ((x1 / V1) - ce)]

    max_evento_min = max([e.fin_min for e in datos.eventos]) if datos.eventos else 60
    minutos_totales = max(datos.minutos_simulacion, int(max_evento_min) + 60)
    t_sim = np.linspace(0, float(minutos_totales), (minutos_totales * 10) + 1)
    
    sol = odeint(ode_sys, [0.0, 0.0, 0.0, 0.0], t_sim)
    Cp = np.clip(sol[:, 0] / V1, 0.0, None)
    Ce = np.clip(sol[:, 3], 0.0, None)

    # === ALERTAS CLÍNICAS ===
    torax_lenoso, depresion_resp, apnea = np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool)
    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina':
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
    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina':
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
            pd_arrays['Frecuencia Mediana EEG (Schüttler 1987)'] = FarmacoMatematica.ketamina_pd(Cp, 'Frecuencia Mediana EEG (Schüttler 1987)').tolist()
            pd_arrays['Presión Arterial Sistólica (Abuhelwa 2022)'] = FarmacoMatematica.ketamina_pd(Cp, 'Presión Arterial Sistólica (Abuhelwa 2022)').tolist()
            pd_arrays['Frecuencia Cardíaca (Abuhelwa 2022)'] = FarmacoMatematica.ketamina_pd(Cp, 'Frecuencia Cardíaca (Abuhelwa 2022)').tolist()
            pd_arrays['Intensidad Disociativa (Olofsen 2022)'] = FarmacoMatematica.ketamina_pd(Ce, 'Intensidad Disociativa (Olofsen 2022)').tolist()
            pd_arrays['Factor de Tolerancia al Estimulo Mecanico (Olofsen 2022)'] = FarmacoMatematica.ketamina_pd(Ce, 'Factor de Tolerancia al Estimulo Mecanico (Olofsen 2022)').tolist()
            pd_arrays['Nivel de Sedación CHWSS (Herd 2008)'] = FarmacoMatematica.herd_2008_sedacion(Ce).tolist()
            pd_arrays['Probabilidad de llegar al despertar (Herd 2008)'] = FarmacoMatematica.herd_2008_prob_despertar(Ce).tolist()
        else: # Remifentanilo
            pd_arrays['Frecuencia Borde Espectral (Minto 1997)'] = FarmacoMatematica.hill(Ce, max(0.1, 13.1 - 0.148 * (datos.edad_anos - 40)), 2.44, 20.0, 5.5).tolist()
            pd_arrays['Frecuencia Borde Espectral (Egan 1996)'] = FarmacoMatematica.hill(Ce, 19.9, 4.3, 19.0, 5.2).tolist()
            pd_arrays['Frecuencia Borde Espectral (Eleveld 2017)'] = FarmacoMatematica.hill(Ce, 12.7, 2.87, 19.9, 5.66).tolist()
            pd_arrays['Factor Tolerancia al Estimulo Mecanico (Abad 2022)'] = FarmacoMatematica.hill(Ce, 2.8, 1.9, 1.0, 3.88).tolist()

    # === INTERACCIONES PD (ISOBOLAS Y 3D) ===
    isobolas_clasicas = {}
    superficie_3d = {}
    grafico_dinamico_iso = {"activo": False}

    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina':
        base_max = max(np.max(Cp), np.max(Ce)) * 1.1 if max(np.max(Cp), np.max(Ce)) > 0 else 10.0
        if datos.farmaco == 'Ketamina': max_y = max(base_max, 1000.0)
        else: max_y = max(base_max, 15.0)
        
        # 1. MODO CLÁSICO (100 puntos en Y fijos)
        ce_y_clasico = np.linspace(0, max_y, 100)
        isobolas_clasicas['ce_farmaco'] = ce_y_clasico.tolist()
        
        # Generar las isóbolas clásicas
        isobolas_tmp = {}
        for c_y in [ce_y_clasico]:
            if datos.farmaco == 'Fentanilo':
                b_50 = max(1.0, 4.9 - 0.09*(datos.edad_anos - 20)); b_95 = b_50*(5.4/3.3); fr_sm = (0.50*c_y)/(0.75+c_y)
                isobolas_tmp['Propofol LOC (Smith 1994)'] = {'ce_50': b_50*(1-fr_sm), 'ce_95': b_95*(1-fr_sm), 'base_50': b_50, 'base_95': b_95, 'unit': 'ug/ml'}
                fr_som = (0.95*c_y**1.4)/(0.63**1.4+c_y**1.4)
                isobolas_tmp['Propofol No Movimiento (Smith 1994)'] = {'ce_50': 15.2*(1-fr_som), 'ce_95': 27.4*(1-fr_som), 'base_50': 15.2, 'base_95': 27.4, 'unit': 'ug/ml'}
                
                eff_map_kazama = {
                    'Propofol ↓ FC 15% (Kazama 1998)': 'Disminución FC 15%', 'Propofol ↓ FC 30% (Kazama 1998)': 'Disminución FC 30%',
                    'Propofol No Movimiento (Kazama 1998)': 'Respuesta Somática', 'Propofol No ↑ PAS 15% (Kazama 1998)': 'Supresión Aumento PAS 15%',
                    'Propofol ↓ PAS 15% (Kazama 1998)': 'Disminución PAS 15%', 'Propofol ↓ PAS 30% (Kazama 1998)': 'Disminución PAS 30%',
                    'Propofol ↓ PAS 40% (Kazama 1998)': 'Disminución PAS 40%'
                }
                for ui_name, eff in eff_map_kazama.items():
                    isobolas_tmp[ui_name] = {'ce_50': FarmacoMatematica.kazama_1998_iso_cprop(c_y, 0.50, eff), 'ce_95': FarmacoMatematica.kazama_1998_iso_cprop(c_y, 0.95, eff), 'base_50': FarmacoMatematica.kazama_1998_iso_cprop(0.0, 0.50, eff), 'base_95': FarmacoMatematica.kazama_1998_iso_cprop(0.0, 0.95, eff), 'unit': 'ug/ml'}
                
                eff_map_katoh = {
                    'Sevoflurano LOC (Katoh 1998)': 'LOC 1998', 'Sevoflurano No ↑ FC 15% / ↑ MAP 15% (Katoh 1999)': 'Simpática 1999',
                    'Sevoflurano No Movimiento (Katoh 1998)': 'Somática 1998', 'Sevoflurano No Movimiento (Katoh 1999)': 'Somática 1999'
                }
                for ui_name, eff in eff_map_katoh.items():
                    b50 = 0.62 if 'LOC' in eff else (1.84 if 'Somática 1998' == eff else (1.85 if 'Somática 1999' == eff else 4.15))
                    b95 = 0.71 if 'LOC' in eff else (2.21 if 'Somática 1998' == eff else (2.28 if 'Somática 1999' == eff else 6.26))
                    isobolas_tmp[ui_name] = {'ce_50': FarmacoMatematica.katoh_iso_csevo(c_y, 0.50, eff), 'ce_95': FarmacoMatematica.katoh_iso_csevo(c_y, 0.95, eff), 'base_50': b50, 'base_95': b95, 'unit': '%'}
                
                isobolas_tmp['Sevoflurano No Movimiento (Vereecke 2013)'] = {'ce_50': FarmacoMatematica.vereecke_iso_cs(c_y, 1.73, 2.07, 0.931, 6.40, 0.50), 'ce_95': FarmacoMatematica.vereecke_iso_cs(c_y, 1.73, 2.07, 0.931, 6.40, 0.95), 'base_50': 1.73, 'base_95': (19.0**(1.0/6.40))*1.73, 'unit': '%'}
                isobolas_tmp['Sevoflurano No ↑ FC 15% / ↑ PAS 15% (Vereecke 2013)'] = {'ce_50': FarmacoMatematica.vereecke_iso_cs(c_y, 4.60, 0.43, 0.931, 6.40, 0.50), 'ce_95': FarmacoMatematica.vereecke_iso_cs(c_y, 4.60, 0.43, 0.931, 6.40, 0.95), 'base_50': 4.60, 'base_95': (19.0**(1.0/6.40))*4.60, 'unit': '%'}
                isobolas_tmp['Propofol Tolerancia LMA ΔANI < 20% (Wang 2026)'] = {'ce_50': FarmacoMatematica.wang_2026_iso_cm(c_y, 9.77, 0.50), 'ce_95': FarmacoMatematica.wang_2026_iso_cm(c_y, 9.77, 0.95), 'base_50': 9.77, 'base_95': 9.77, 'unit': 'ug/ml'}
                isobolas_tmp['Sevoflurano Tolerancia LMA ΔANI < 20% (Wang 2026)'] = {'ce_50': FarmacoMatematica.wang_2026_iso_cm(c_y, 4.29, 0.50), 'ce_95': FarmacoMatematica.wang_2026_iso_cm(c_y, 4.29, 0.95), 'base_50': 4.29, 'base_95': 4.29, 'unit': '%'}
            
            elif datos.farmaco == 'Remifentanilo':
                isobolas_tmp['Ketamina Probabilidad de Conservar la Ventilación Minuto (Jonkman 2018)'] = {'ce_50': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.50, solve_for='ket'), 'ce_95': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.80, solve_for='ket'), 'base_50': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.50, 'ket'), 'base_95': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.80, 'ket'), 'unit': 'ng/ml'}
                isobolas_tmp['Propofol BIS (Bouillon 2004)'] = {'ce_50': FarmacoMatematica.bouillon_2004_bis_iso_cprop(c_y, 60.0), 'ce_95': FarmacoMatematica.bouillon_2004_bis_iso_cprop(c_y, 40.0), 'base_50': FarmacoMatematica.bouillon_2004_bis_iso_cprop(0.0, 60.0), 'base_95': FarmacoMatematica.bouillon_2004_bis_iso_cprop(0.0, 40.0), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol LOC (Bouillon 2004)'] = {'ce_50': FarmacoMatematica.bouillon_2004_iso_cprop(c_y, 0.50, 'LOC'), 'ce_95': FarmacoMatematica.bouillon_2004_iso_cprop(c_y, 0.95, 'LOC'), 'base_50': FarmacoMatematica.bouillon_2004_iso_cprop(0.0, 0.50, 'LOC'), 'base_95': FarmacoMatematica.bouillon_2004_iso_cprop(0.0, 0.95, 'LOC'), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol Tolerancia Laringoscopia (Bouillon 2004)'] = {'ce_50': FarmacoMatematica.bouillon_2004_iso_cprop(c_y, 0.50, 'Laringoscopia'), 'ce_95': FarmacoMatematica.bouillon_2004_iso_cprop(c_y, 0.95, 'Laringoscopia'), 'base_50': FarmacoMatematica.bouillon_2004_iso_cprop(0.0, 0.50, 'Laringoscopia'), 'base_95': FarmacoMatematica.bouillon_2004_iso_cprop(0.0, 0.95, 'Laringoscopia'), 'unit': 'ug/ml'}
                
                isobolas_tmp['Propofol PSI-2 (Kuizenga 2019)'] = {'ce_50': FarmacoMatematica.kuizenga_2019_iso_c('Propofol', 'PSI', c_y, 50.0), 'ce_95': FarmacoMatematica.kuizenga_2019_iso_c('Propofol', 'PSI', c_y, 25.0), 'base_50': FarmacoMatematica.kuizenga_2019_iso_c('Propofol', 'PSI', 0.0, 50.0), 'base_95': FarmacoMatematica.kuizenga_2019_iso_c('Propofol', 'PSI', 0.0, 25.0), 'unit': 'ug/ml'}
                isobolas_tmp['Sevoflurano PSI-2 (Kuizenga 2019)'] = {'ce_50': FarmacoMatematica.kuizenga_2019_iso_c('Sevoflurano', 'PSI', c_y, 50.0), 'ce_95': FarmacoMatematica.kuizenga_2019_iso_c('Sevoflurano', 'PSI', c_y, 25.0), 'base_50': FarmacoMatematica.kuizenga_2019_iso_c('Sevoflurano', 'PSI', 0.0, 50.0), 'base_95': FarmacoMatematica.kuizenga_2019_iso_c('Sevoflurano', 'PSI', 0.0, 25.0), 'unit': '%'}

                p_greco = {
                    'Propofol MOAA/S ≤ 1 (Johnson 2008)': (2.2, 33.1, 3.6, 5.0, False), 'Propofol MOAA/S ≤ 3 (Kern 2004)': (1.8, 12.5, 5.1, 5.8, False),
                    'Propofol MOAA/S ≥ 2 (Johnson 2008)': (1.3, 10.5, 2.8, 3.5, True), 'Propofol MOAA/S ≥ 4 (Kern 2004)': (1.8, 12.5, 5.1, 5.8, True),
                    'Propofol Tolerancia al Estimulo Electrico (Kern 2004)': (4.56, 21.3, 14.7, 6.0, False), 'Propofol Tolerancia al Estimulo Mecanico (Kern 2004)': (4.16, 8.84, 8.2, 8.3, False),
                    'Propofol Tolerancia Laringoscopia (Kern 2004)': (5.60, 48.9, 33.2, 2.2, False), 'Sevoflurano MOAA/S ≤ 1 (Johnson 2010)': (0.74, 50.9, 9.4, 5.2, False),
                    'Sevoflurano MOAA/S ≥ 2 (Johnson 2010)': (0.74, 50.9, 9.4, 5.2, True), 'Sevoflurano Tolerancia al estimulo mecanico (Johnson 2010)': (0.83, 1.3, 0.9, 2.7, False)
                }
                for k, v in p_greco.items():
                    k50, k95 = (50.0/50.0)**(1.0/v[3]), ((5.0/95.0)**(1.0/v[3]) if v[4] else (95.0/5.0)**(1.0/v[3]))
                    isobolas_tmp[k] = {'ce_50': FarmacoMatematica.greco_iso_cs(c_y, v[0], v[1], v[2], v[3], 0.50, v[4]), 'ce_95': FarmacoMatematica.greco_iso_cs(c_y, v[0], v[1], v[2], v[3], 0.95, v[4]), 'base_50': v[0]*k50, 'base_95': v[0]*k95, 'unit': 'ug/ml' if 'Propofol' in k else '%'}
                
                isobolas_tmp["Sevoflurano BIS (Manyam 2007)"] = {'ce_50': FarmacoMatematica.greco_iso_cs(c_y, 2.37, 38.02, 0.52, 1.12, 0.60, True), 'ce_95': FarmacoMatematica.greco_iso_cs(c_y, 2.37, 38.02, 0.52, 1.12, 0.40, True), 'base_50': 2.37*((100.0/60 - 1.0)**(1.0/1.12)), 'base_95': 2.37*((100.0/40 - 1.0)**(1.0/1.12)), 'unit': '%'}

                p_man = {
                    'Sevoflurano MOAA/S ≤ 1 (Manyam 2006)': (7.30, 7.84, 0.23, 3.94), 'Sevoflurano MOAA/S ≥ 4 (Manyam 2006)': (4.19, 4.25, 0.28, 0.58),
                    'Sevoflurano Tolerancia al estimulo mecanico (Manyam 2006)': (3.82, 2.43, 0.54, 1.27), 'Sevoflurano Tolerancia al estimulo termico (Manyam 2006)': (3.38, 1.32, 0.55, 3.47),
                    'Sevoflurano Tolerancia al estimulo electrico (Manyam 2006)': (3.27, 0.97, 0.088, 1.09), 'Sevoflurano Tolerancia Laringoscopia (Manyam 2006)': (3.70, 2.36, 0.54, 1.22)
                }
                for k, v in p_man.items():
                    isobolas_tmp[k] = {'ce_50': FarmacoMatematica.manyam_iso_cs(c_y, v[0], v[1], v[2], v[3], 0.50), 'ce_95': FarmacoMatematica.manyam_iso_cs(c_y, v[0], v[1], v[2], v[3], 0.95), 'base_50': v[0]/v[1], 'base_95': (v[0]+2.9444)/v[1], 'unit': '%'}

                e0_bis, gam_bis, c50_r_bis, c50_s_bisp, c50_s_biso = 89.5, 1.88, 27.3, 2.29, 1.99
                isobolas_tmp['Sevoflurano BIS post-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_bis, c50_s_bisp, c50_r_bis, gam_bis), 'unit': '%'}
                isobolas_tmp['Sevoflurano BIS pre-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_bis, c50_s_biso, c50_r_bis, gam_bis), 'unit': '%'}
                
                e0_se, gam_se, c50_r_se, c50_s_sep, c50_s_seo = 97.1, 1.87, 16.2, 2.13, 1.82
                isobolas_tmp['Sevoflurano SE post-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_se, c50_s_sep, c50_r_se, gam_se), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_se, c50_s_sep, c50_r_se, gam_se), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_se, c50_s_sep, c50_r_se, gam_se), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_se, c50_s_sep, c50_r_se, gam_se), 'unit': '%'}
                isobolas_tmp['Sevoflurano SE pre-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_se, c50_s_seo, c50_r_se, gam_se), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_se, c50_s_seo, c50_r_se, gam_se), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_se, c50_s_seo, c50_r_se, gam_se), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_se, c50_s_seo, c50_r_se, gam_se), 'unit': '%'}
                
                e0_re, gam_re, c50_r_re, c50_s_rep, c50_s_reo = 103.0, 2.08, 18.2, 2.24, 1.88
                isobolas_tmp['Sevoflurano RE post-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'unit': '%'}
                isobolas_tmp['Sevoflurano RE pre-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'unit': '%'}

            elif datos.farmaco == 'Ketamina':
                isobolas_tmp['Remifentanilo Probabilidad de Conservar la Ventilación Minuto (Jonkman 2018)'] = {'ce_50': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.50, solve_for='remi'), 'ce_95': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.80, solve_for='remi'), 'base_50': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.50, 'remi'), 'base_95': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.80, 'remi'), 'unit': 'ng/ml'}
                isobolas_tmp['Propofol LOC (Hui 1995)'] = {'ce_50': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.50, efecto='LOC', solve_for='prop'), 'ce_95': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.95, efecto='LOC', solve_for='prop'), 'base_50': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.50, efecto='LOC', solve_for='prop'), 'base_95': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.95, efecto='LOC', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol Tolerancia al Estímulo Eléctrico (Hui 1995)'] = {'ce_50': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.50, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'ce_95': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.95, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'base_50': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.50, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'base_95': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.95, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Sevoflurano No Movimiento (Hamp 2018)'] = {'ce_50': FarmacoMatematica.hamp_2018_iso_csevo(c_y, 0.50), 'ce_95': np.full_like(c_y, np.nan), 'base_50': 2.2, 'base_95': np.nan, 'unit': '%'}

        for k, v in isobolas_tmp.items():
            isobolas_clasicas[k] = {'ce_50': clean_arr(v['ce_50']), 'ce_95': clean_arr(v['ce_95'])}
            
        # 2. MODO DINÁMICO (Evaluado sobre el arreglo Ce real en el tiempo)
        if datos.isobolograma_2d != "Ninguno" and datos.isobolograma_2d in isobolas_tmp and datos.tipo_grafico_iso != "Clásico":
            req_dict = isobolas_tmp[datos.isobolograma_2d]
            ce_dinamico = Ce # Array completo (14401 pts)
            
            # Recalculamos los requerimientos exactos pero sobre Ce dinámico
            if datos.farmaco == 'Fentanilo':
                if 'Smith 1994' in datos.isobolograma_2d:
                    b_50 = req_dict['base_50']; b_95 = req_dict['base_95']
                    p_val, g_val, c50_f = (0.50, 1.0, 0.75) if 'LOC' in datos.isobolograma_2d else (0.95, 1.4, 0.63)
                    fr_y = (p_val*ce_dinamico**g_val)/(c50_f**g_val+ce_dinamico**g_val)
                    req50_arr = b_50*(1-fr_y); req95_arr = b_95*(1-fr_y)
                elif 'Kazama 1998' in datos.isobolograma_2d:
                    eff = next((v for k, v in eff_map_kazama.items() if k in datos.isobolograma_2d), 'Respuesta Somática')
                    req50_arr = FarmacoMatematica.kazama_1998_iso_cprop(ce_dinamico, 0.50, eff)
                    req95_arr = FarmacoMatematica.kazama_1998_iso_cprop(ce_dinamico, 0.95, eff)
                elif 'Katoh' in datos.isobolograma_2d:
                    eff = next((v for k, v in eff_map_katoh.items() if k in datos.isobolograma_2d), 'LOC 1998')
                    req50_arr = FarmacoMatematica.katoh_iso_csevo(ce_dinamico, 0.50, eff)
                    req95_arr = FarmacoMatematica.katoh_iso_csevo(ce_dinamico, 0.95, eff)
                elif 'Vereecke 2013' in datos.isobolograma_2d:
                    is_mov = 'Movimiento' in datos.isobolograma_2d; c50_f = 2.07 if is_mov else 0.43; c50_s = 1.73 if is_mov else 4.60
                    req50_arr = FarmacoMatematica.vereecke_iso_cs(ce_dinamico, c50_s, c50_f, 0.931, 6.40, 0.50)
                    req95_arr = FarmacoMatematica.vereecke_iso_cs(ce_dinamico, c50_s, c50_f, 0.931, 6.40, 0.95)
                elif 'Wang 2026' in datos.isobolograma_2d:
                    is_prop = 'Propofol' in datos.isobolograma_2d; c50_m = 9.77 if is_prop else 4.29
                    req50_arr = FarmacoMatematica.wang_2026_iso_cm(ce_dinamico, c50_m, 0.50)
                    req95_arr = FarmacoMatematica.wang_2026_iso_cm(ce_dinamico, c50_m, 0.95)
            
            elif datos.farmaco == 'Remifentanilo':
                if 'Jonkman' in datos.isobolograma_2d:
                    req50_arr = FarmacoMatematica.jonkman_2018_iso_cx(ce_dinamico, 0.50, 'ket')
                    req95_arr = FarmacoMatematica.jonkman_2018_iso_cx(ce_dinamico, 0.80, 'ket')
                elif 'Bouillon 2004' in datos.isobolograma_2d:
                    if 'BIS' in datos.isobolograma_2d:
                        req50_arr = FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_dinamico, 60.0)
                        req95_arr = FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_dinamico, 40.0)
                    else:
                        eff_str = 'LOC' if 'LOC' in datos.isobolograma_2d else 'Laringoscopia'
                        req50_arr = FarmacoMatematica.bouillon_2004_iso_cprop(ce_dinamico, 0.50, eff_str)
                        req95_arr = FarmacoMatematica.bouillon_2004_iso_cprop(ce_dinamico, 0.95, eff_str)
                elif 'Kuizenga 2019' in datos.isobolograma_2d:
                    f_str = 'Propofol' if 'Propofol' in datos.isobolograma_2d else 'Sevoflurano'
                    req50_arr = FarmacoMatematica.kuizenga_2019_iso_c(f_str, 'PSI', ce_dinamico, 50.0)
                    req95_arr = FarmacoMatematica.kuizenga_2019_iso_c(f_str, 'PSI', ce_dinamico, 25.0)
                elif 'Kern 2004' in datos.isobolograma_2d or 'Johnson' in datos.isobolograma_2d:
                    v = p_greco[datos.isobolograma_2d]
                    req50_arr = FarmacoMatematica.greco_iso_cs(ce_dinamico, v[0], v[1], v[2], v[3], 0.50, v[4])
                    req95_arr = FarmacoMatematica.greco_iso_cs(ce_dinamico, v[0], v[1], v[2], v[3], 0.95, v[4])
                elif 'Manyam 2007' in datos.isobolograma_2d:
                    req50_arr = FarmacoMatematica.greco_iso_cs(ce_dinamico, 2.37, 38.02, 0.52, 1.12, 0.60, True)
                    req95_arr = FarmacoMatematica.greco_iso_cs(ce_dinamico, 2.37, 38.02, 0.52, 1.12, 0.40, True)
                elif 'Manyam 2006' in datos.isobolograma_2d:
                    v = p_man[datos.isobolograma_2d]
                    req50_arr = FarmacoMatematica.manyam_iso_cs(ce_dinamico, v[0], v[1], v[2], v[3], 0.50)
                    req95_arr = FarmacoMatematica.manyam_iso_cs(ce_dinamico, v[0], v[1], v[2], v[3], 0.95)
                elif 'Heyse 2014' in datos.isobolograma_2d:
                    if 'BIS' in datos.isobolograma_2d: e0, gamma, c50_r, c50_s = 89.5, 1.88, 27.3, (2.29 if 'post' in datos.isobolograma_2d else 1.99)
                    elif 'SE ' in datos.isobolograma_2d: e0, gamma, c50_r, c50_s = 97.1, 1.87, 16.2, (2.13 if 'post' in datos.isobolograma_2d else 1.82)
                    else: e0, gamma, c50_r, c50_s = 103.0, 2.08, 18.2, (2.24 if 'post' in datos.isobolograma_2d else 1.88)
                    req50_arr = FarmacoMatematica.heyse_2014_iso_csevo(ce_dinamico, 60.0, e0, c50_s, c50_r, gamma)
                    req95_arr = FarmacoMatematica.heyse_2014_iso_csevo(ce_dinamico, 40.0, e0, c50_s, c50_r, gamma)
            
            elif datos.farmaco == 'Ketamina':
                if 'Jonkman 2018' in datos.isobolograma_2d:
                    req50_arr = FarmacoMatematica.jonkman_2018_iso_cx(ce_dinamico, 0.50, 'remi')
                    req95_arr = FarmacoMatematica.jonkman_2018_iso_cx(ce_dinamico, 0.80, 'remi')
                elif 'Hui 1995' in datos.isobolograma_2d:
                    ef_str = 'Tolerancia al Estímulo Eléctrico' if 'Eléctrico' in datos.isobolograma_2d else 'LOC'
                    req50_arr = FarmacoMatematica.hui_1995_iso_cx(ce_dinamico, 0.50, ef_str, 'prop')
                    req95_arr = FarmacoMatematica.hui_1995_iso_cx(ce_dinamico, 0.95, ef_str, 'prop')
                elif 'Hamp 2018' in datos.isobolograma_2d:
                    req50_arr = FarmacoMatematica.hamp_2018_iso_csevo(ce_dinamico, 0.50)
                    req95_arr = np.full_like(ce_dinamico, np.nan)

            # Generamos las curvas para la app móvil
            curvas = []
            nombre_droga = datos.isobolograma_2d.split(' ')[0]
            unit = req_dict['unit']

            if 'Requerimiento' in datos.tipo_grafico_iso:
                curvas.append({"nombre": f"Req. 50% ({unit})", "color": "#ba68c8", "es_linea_punteada": True, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(req50_arr)})
                curvas.append({"nombre": f"Req. 95% ({unit})", "color": "#7b1fa2", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(req95_arr)})
                
                # Eje secundario EEG Predictivo (Solo para Remifentanilo)
                if datos.farmaco == 'Remifentanilo' and any(kw in datos.tipo_grafico_iso for kw in ['BIS', 'PSI', 'SE ', 'RE ']):
                    is_prop = 'Propofol' in datos.isobolograma_2d
                    f_str = 'Propofol' if is_prop else 'Sevoflurano'
                    
                    if 'PSI' in datos.tipo_grafico_iso:
                        bis50_arr = FarmacoMatematica.kuizenga_2019_prob(req50_arr, f_str, 'PSI', ce_dinamico)
                        bis95_arr = FarmacoMatematica.kuizenga_2019_prob(req95_arr, f_str, 'PSI', ce_dinamico)
                        lbl_eeg = "PSI"
                    elif is_prop and 'BIS' in datos.tipo_grafico_iso:
                        bis50_arr = FarmacoMatematica.bouillon_2004_bis(req50_arr, ce_dinamico)
                        bis95_arr = FarmacoMatematica.bouillon_2004_bis(req95_arr, ce_dinamico)
                        lbl_eeg = "BIS"
                    elif not is_prop:
                        if 'Requerimiento Dinámico BIS' == datos.tipo_grafico_iso:
                            bis50_arr = FarmacoMatematica.estimar_bis_global(req50_arr, ce_dinamico)
                            bis95_arr = FarmacoMatematica.estimar_bis_global(req95_arr, ce_dinamico)
                            lbl_eeg = "BIS"
                        else:
                            e0, gamma, c50_r = 89.5, 1.88, 27.3; lbl_eeg = "BIS"
                            if 'SE ' in datos.tipo_grafico_iso: e0, gamma, c50_r, lbl_eeg = 97.1, 1.87, 16.2, "SE"
                            elif 'RE ' in datos.tipo_grafico_iso: e0, gamma, c50_r, lbl_eeg = 103.0, 2.08, 18.2, "RE"
                            
                            is_post = 'post' in datos.tipo_grafico_iso
                            if 'BIS' in datos.tipo_grafico_iso: c50_s = 2.29 if is_post else 1.99
                            elif 'SE ' in datos.tipo_grafico_iso: c50_s = 2.13 if is_post else 1.82
                            elif 'RE ' in datos.tipo_grafico_iso: c50_s = 2.24 if is_post else 1.88
                            else: c50_s = 1.99
                            
                            bis50_arr = FarmacoMatematica.heyse_2014_prob(req50_arr, ce_dinamico, e0, c50_s, c50_r, gamma)
                            bis95_arr = FarmacoMatematica.heyse_2014_prob(req95_arr, ce_dinamico, e0, c50_s, c50_r, gamma)

                    curvas.append({"nombre": f"{lbl_eeg} si Req. 50%", "color": "#4dd0e1", "es_linea_punteada": True, "es_eje_secundario": True, "es_area": False, "datos": clean_arr(bis50_arr)})
                    curvas.append({"nombre": f"{lbl_eeg} si Req. 95%", "color": "#00acc1", "es_linea_punteada": False, "es_eje_secundario": True, "es_area": False, "datos": clean_arr(bis95_arr)})

            elif 'Ahorro' in datos.tipo_grafico_iso:
                ahorro_50 = (1.0 - req50_arr / req_dict['base_50']) * 100.0 if req_dict['base_50'] > 0 else np.zeros_like(ce_dinamico)
                ahorro_95 = (1.0 - req95_arr / req_dict['base_95']) * 100.0 if req_dict['base_95'] > 0 else np.zeros_like(ce_dinamico)
                curvas.append({"nombre": f"Ahorro 50% (%)", "color": "#ba68c8", "es_linea_punteada": True, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(ahorro_50)})
                curvas.append({"nombre": f"Ahorro 95% (%)", "color": "#7b1fa2", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(ahorro_95)})

            elif 'Contribución' in datos.tipo_grafico_iso:
                is_95 = '95%' in datos.tipo_grafico_iso
                req_arr = req95_arr if is_95 else req50_arr
                base_val = req_dict['base_95'] if is_95 else req_dict['base_50']
                if base_val > 0:
                    ahorro_arr = np.clip((1.0 - req_arr / base_val) * 100.0, 0, 100)
                else:
                    ahorro_arr = np.zeros_like(ce_dinamico)
                curvas.append({"nombre": f"Aporte {datos.farmaco} (%)", "color": "#4caf50", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": True, "datos": clean_arr(ahorro_arr)})
                curvas.append({"nombre": f"Aporte {nombre_droga} (%)", "color": "#ff9800", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": True, "datos": clean_arr(np.full_like(ce_dinamico, 100.0))})

            grafico_dinamico_iso = {
                "activo": True,
                "titulo_grafico": f"{datos.tipo_grafico_iso} - {datos.isobolograma_2d}",
                "etiqueta_y": f"{nombre_droga} ({unit})" if 'Requerimiento' in datos.tipo_grafico_iso else "Porcentaje (%)",
                "etiqueta_y2": "EEG Proyectado" if 'Requerimiento' in datos.tipo_grafico_iso and 'eje_secundario' in str(curvas) else None,
                "limite_y": 105.0 if 'Requerimiento' not in datos.tipo_grafico_iso else None,
                "curvas": curvas
            }

        # SUPERFICIE 3D UNIFICADA
        if datos.modelo_3d != 'Ninguna' and datos.modelo_3d in isobolas_tmp:
            is_prop = 'Propofol' in datos.modelo_3d
            if datos.farmaco == 'Fentanilo':
                y_max, x_max = 10.0, 30.0 if is_prop else 8.0
                ylab = 'Ce Fentanilo (ng/ml)'
                xlab = 'Ce Propofol (ug/ml)' if is_prop else 'etSEV (%)'
                zlab = datos.modelo_3d
            elif datos.farmaco == 'Remifentanilo':
                if 'Jonkman' in datos.modelo_3d:
                    y_max, x_max = max(base_max, 10.0), 1000.0
                    ylab, xlab, zlab = 'Ce Remifentanilo (ng/ml)', 'Ce Ketamina (ng/ml)', datos.modelo_3d
                else:
                    y_max, x_max = max(base_max, 10.0), 15.0 if is_prop else 8.0
                    ylab = 'Ce Remifentanilo (ng/ml)'
                    xlab = 'Ce Propofol (ug/ml)' if is_prop else 'etSEV (%)'
                    zlab = datos.modelo_3d
            else: # Ketamina
                if 'Jonkman' in datos.modelo_3d:
                    y_max, x_max = max(base_max, 1000.0), 10.0
                    ylab, xlab, zlab = 'Ce Ketamina (ng/ml)', 'Ce Remifentanilo (ng/ml)', datos.modelo_3d
                elif 'Hui' in datos.modelo_3d:
                    y_max, x_max = max(base_max, 1000.0), 15.0
                    ylab, xlab, zlab = 'Ce Ketamina (ng/ml)', 'Ce Propofol (ug/ml)', datos.modelo_3d
                else:
                    y_max, x_max = max(base_max, 1000.0), 15.0
                    ylab, xlab, zlab = 'Ce Ketamina (ng/ml)', 'Otro Agente', datos.modelo_3d
                
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
                elif 'Kuizenga 2019' in datos.modelo_3d:
                    f_str = 'Propofol' if 'Propofol' in datos.modelo_3d else 'Sevoflurano'
                    Z = FarmacoMatematica.kuizenga_2019_prob(X, f_str, 'PSI', Y)
                elif 'Jonkman' in datos.modelo_3d:
                    Z = FarmacoMatematica.jonkman_2018_prob(Y, X) # Y=Remi, X=Ketamina
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
            elif datos.farmaco == 'Ketamina':
                if 'Jonkman' in datos.modelo_3d:
                    Z = FarmacoMatematica.jonkman_2018_prob(X, Y) # X=Remi, Y=Ketamina
                elif 'Hui' in datos.modelo_3d:
                    Z = FarmacoMatematica.hui_1995_prob(X, Y, efecto='Tolerancia al Estímulo Eléctrico' if 'Eléctrico' in datos.modelo_3d else 'LOC', as_percent=True)
                else: 
                    Z = np.zeros_like(X)

            Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=0.0)
            superficie_3d = {
                "x_mesh": x_m.tolist(), "y_mesh": y_m.tolist(), "z_mesh": np.round(Z, 2).tolist(),
                "iso50_x": clean_arr(isobolas_clasicas[datos.modelo_3d]['ce_50']), "iso50_y": clean_arr(isobolas_clasicas['ce_farmaco']),
                "iso80_x": clean_arr(isobolas_clasicas[datos.modelo_3d].get('ce_80', [])),
                "iso95_x": clean_arr(isobolas_clasicas[datos.modelo_3d].get('ce_95', [])), "iso95_y": clean_arr(isobolas_clasicas['ce_farmaco']),
                "xlabel": xlab, "ylabel": ylab, "zlabel": zlab, "title": datos.modelo_3d
            }

    return {
        "tiempo_minutos": np.round(t_sim, 2).tolist(),
        "cp": np.round(Cp, 3).tolist(), "ce": np.round(Ce, 3).tolist(),
        "alertas_clinicas": {"torax_lenoso": torax_lenoso.tolist(), "depresion_respiratoria": depresion_resp.tolist(), "apnea": apnea.tolist()},
        "farmacodinamia_pd": {k: np.round(v, 2).tolist() for k, v in pd_arrays.items()},
        "isobolas_interaccion": isobolas_clasicas,
        "grafico_dinamico_iso": grafico_dinamico_iso,
        "superficie_3d": superficie_3d
    }
