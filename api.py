from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

# 1. ESTRUCTURAS DE ENTRADA
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
    modelo_3d: str = "Ninguna"  # <-- NUEVO: Recibe el modelo 3D desde Flutter

app = FastAPI(title="TIVA Flow API Motor Matemático")

# 2. CLASE MATEMÁTICA PURA (Farmacodinamia e Isobolas)
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
    def wang_2026_iso_cm(cf, c50_m, target_prob):
        def wang_prob(cm, cf_val):
            u_safe = np.where((cm/c50_m + cf_val/2.58) == 0, 1e-6, (cm/c50_m + cf_val/2.58))
            x, y = (cm/c50_m)/u_safe, (cf_val/2.58)/u_safe
            U50 = 10.0**((1.0-x)*(1.0-y)*(-0.06*x - 1.35*y - 0.78*x*y))
            n = 1.98*x + 1.98*y + 1.44*x*y
            return (((cm/c50_m + cf_val/2.58)/U50)**n) / (1.0 + ((cm/c50_m + cf_val/2.58)/U50)**n)
        
        if isinstance(cf, np.ndarray):
            res = []
            for val in cf:
                try:
                    if wang_prob(30.0, val) - target_prob > 0:
                        res.append(root_scalar(lambda cm: wang_prob(cm, val) - target_prob, bracket=[0, 30]).root)
                    else: res.append(np.nan)
                except: res.append(np.nan)
            return np.array(res)
        return np.nan

    @staticmethod
    def bouillon_2004_iso_cprop(c_remi, target_prob, efecto='Laringoscopia'):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(c_remi, np.nan) if isinstance(c_remi, np.ndarray) else np.nan
        c50_remi, c50_prop, gamma_remi, gamma_prop = 1.07, 8.04, 0.97, 5.1
        i_pre = 0.60 if efecto == 'LOC' else 1.05 
        c_remi_safe = np.clip(c_remi, 1e-6, None) if isinstance(c_remi, np.ndarray) else max(1e-6, c_remi)
        num_remi = c_remi_safe ** gamma_remi
        den_remi = num_remi + (c50_remi * i_pre) ** gamma_remi
        i_post = i_pre * (1.0 - (num_remi / den_remi))
        term = (target_prob / (1.0 - target_prob)) ** (1.0 / gamma_prop)
        return (c50_prop * i_post) * term
        
    @staticmethod
    def bouillon_2004_bis_iso_cprop(c_remi, target_bis):
        e0, emax, c50_prop, c50_remi, gamma = 97.4, 97.4, 4.47, 19.3, 1.43
        if target_bis <= (e0 - emax) or target_bis >= e0: return np.full_like(c_remi, np.nan) if isinstance(c_remi, np.ndarray) else np.nan
        effect_ratio = (e0 - target_bis) / emax
        u_req = (effect_ratio / (1.0 - effect_ratio)) ** (1.0 / gamma)
        c_remi_safe = np.clip(c_remi, 0.0, None) if isinstance(c_remi, np.ndarray) else max(0.0, c_remi)
        c_prop = c50_prop * (u_req - (c_remi_safe / c50_remi))
        if isinstance(c_prop, np.ndarray): c_prop[c_prop < 0] = np.nan
        else: 
            if c_prop < 0: return np.nan
        return c_prop

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
    def katoh_1999_iso_csevo(c_fent, target_prob, efecto):
        if target_prob <= 0 or target_prob >= 1: return np.full_like(c_fent, np.nan) if isinstance(c_fent, np.ndarray) else np.nan
        if efecto == 'LOC': base_50, base_95, c50_fent, gamma_fent, max_red = 0.62, 0.71, 7.3, 1.2, 1.0
        elif efecto == 'Respuesta Simpática': base_50, base_95, c50_fent, gamma_fent, max_red = 4.15, 6.26, 0.78, 1.5, 0.95
        elif efecto == 'Respuesta Somática': base_50, base_95, c50_fent, gamma_fent, max_red = 1.77, 2.21, 1.08, 1.0, 0.80
        else: return np.full_like(c_fent, np.nan) if isinstance(c_fent, np.ndarray) else np.nan
        fr = (max_red * c_fent**gamma_fent) / (c50_fent**gamma_fent + c_fent**gamma_fent)
        u50 = np.clip(base_50 * (1.0 - fr), 1e-5, None) if isinstance(base_50 * (1.0 - fr), np.ndarray) else max(1e-5, base_50 * (1.0 - fr))
        ratio = np.clip((base_95 * (1.0 - fr)) / u50, 1.01, None) if isinstance((base_95 * (1.0 - fr)) / u50, np.ndarray) else max(1.01, (base_95 * (1.0 - fr)) / u50)
        gamma = np.log(19.0) / np.log(ratio)
        return u50 * ((target_prob / (1.0 - target_prob))**(1.0 / gamma))

# 3. FARMACOCINÉTICA
def calcular_ffm(peso, altura, sexo, edad):
    bmi = peso / ((max(1.0, altura) / 100.0) ** 2)
    wbm = (9270.0 * peso) / (6680.0 + 216.0 * bmi) if sexo == 'Masculino' else (9270.0 * peso) / (8780.0 + 244.0 * bmi)
    a_mat, A50, c_mat = (0.88, 13.4, 12.7) if sexo == 'Masculino' else (1.11, 7.1, 1.1)
    mf = a_mat + ((1.0 - a_mat) * (edad / A50)**c_mat) / (1.0 + (edad / A50)**c_mat) if edad > 0 else 1.0
    return mf * wbm

def get_pk_params(farmaco, modelo_pk, ke0_tpeak_str, peso, altura, sexo, edad):
    if farmaco == 'Fentanilo':
        if 'Scott 1987' in modelo_pk: V1, V2, V3 = 12.7, 50.7, 274.0; Cl1, Cl2, Cl3 = 0.574, 4.01, 1.95
        elif 'Shafer 1990 (Fijo)' in modelo_pk: V1, V2, V3 = 6.09, 28.1, 228.0; Cl1, Cl2, Cl3 = 0.504, 2.87, 1.37
        elif 'Shafer 1990 (Peso' in modelo_pk: V1, V2, V3 = 0.105*peso, 0.446*peso, 3.37*peso; Cl1, Cl2, Cl3 = 0.00838*peso, 0.0474*peso, 0.0199*peso
        elif 'Bae 2020' in modelo_pk: f_vol, f_cl = (peso/70.0)**1.23, (peso/70.0)**0.313; V1, V2, V3 = 10.1*f_vol, 26.5*f_vol, 206.0*f_vol; Cl1, Cl2, Cl3 = 0.704*f_cl, 2.38*f_cl, 1.49*f_cl
        elif 'Ginsberg' in modelo_pk: V1, V2, V3 = max(0.001, 0.43*(peso-19.8)+5.8), max(0.001, 6.2*(edad-6.4)+34.4), 0.0; Cl1, Cl2, Cl3 = max(0.001, 0.01*(peso-19.8)+0.35), max(0.001, 0.82), 0.0
        elif 'Maharaj' in modelo_pk: V1, V2, V3 = 10.8*(peso/70.0), 417.0*(peso/70.0), 0.0; Cl1, Cl2, Cl3 = (32.5/60.0)*((peso/70.0)**0.75), (104.0/60.0)*((peso/70.0)**0.75), 0.0
        else: V1, V2, V3 = 12.7, 50.7, 274.0; Cl1, Cl2, Cl3 = 0.574, 4.01, 1.95
        
        k10, k12, k21, k13, k31 = Cl1/V1, Cl2/V1, Cl2/V2, Cl3/V1, Cl3/V3
        if 'Pediátrico' in modelo_pk: ke0 = 0.0
        elif '0.108' in ke0_tpeak_str: ke0 = 0.108
        elif '0.105' in ke0_tpeak_str: ke0 = 0.105
        elif '4 min' in ke0_tpeak_str:
            a, b, c = k10+k12+k13+k21+k31, k10*(k21+k31)+k12*k31+k13*k21+k21*k31, k10*k21*k31
            lambdas = sorted(-np.real(np.roots([1, a, b, c])[np.isreal(np.roots([1, a, b, c]))]), reverse=True)
            if len(lambdas) < 3: ke0 = 0.147 
            else:
                l1, l2, l3 = lambdas[:3]
                A = (k21-l1)*(k31-l1)/((l2-l1)*(l3-l1)); B = (k21-l2)*(k31-l2)/((l1-l2)*(l3-l2)); C_coeff = (k21-l3)*(k31-l3)/((l1-l3)*(l2-l3))
                def obj(ke): return (A*np.exp(-l1*4.0) + B*np.exp(-l2*4.0) + C_coeff*np.exp(-l3*4.0)) - ke*(A*(4.0*np.exp(-ke*4.0) if abs(ke-l1)<1e-6 else (np.exp(-l1*4.0)-np.exp(-ke*4.0))/(ke-l1)) + B*(4.0*np.exp(-ke*4.0) if abs(ke-l2)<1e-6 else (np.exp(-l2*4.0)-np.exp(-ke*4.0))/(ke-l2)) + C_coeff*(4.0*np.exp(-ke*4.0) if abs(ke-l3)<1e-6 else (np.exp(-l3*4.0)-np.exp(-ke*4.0))/(ke-l3)))
                try: ke0 = root_scalar(obj, bracket=[0.001, 3.0], method='brentq').root
                except: ke0 = 0.147
        else: ke0 = 0.147

    else: # Remifentanilo
        lbm = (1.1*peso - 128.0*((peso/altura)**2)) if sexo == 'Masculino' else (1.07*peso - 148.0*((peso/altura)**2))
        ffm = calcular_ffm(peso, altura, sexo, edad)
        if 'Minto' in modelo_pk: V1, V2, V3 = max(0.01, 5.1-0.0201*(edad-40)+0.072*(lbm-55)), max(0.01, 9.82-0.0811*(edad-40)+0.108*(lbm-55)), 5.42; Cl1, Cl2, Cl3 = max(0.001, 2.6-0.0162*(edad-40)+0.0191*(lbm-55)), max(0.001, 2.05-0.0301*(edad-40)), max(0.001, 0.076-0.00113*(edad-40))
        elif 'La Colla' in modelo_pk: V1, V2, V3 = max(0.01, 5.1-0.0201*(edad-40)+0.072*(ffm-55)), max(0.01, 9.82-0.0811*(edad-40)+0.108*(ffm-55)), 5.42; Cl1, Cl2, Cl3 = max(0.001, 2.6-0.0162*(edad-40)+0.0191*(ffm-55)), max(0.001, 2.05-0.0301*(edad-40)), max(0.001, 0.076-0.00113*(edad-40))
        elif 'Kim' in modelo_pk: V1, V2, V3 = max(0.01, 4.76*((peso/74.5)**0.658)), max(0.01, 8.4*((ffm/52.3)**0.573)-0.0936*(edad-37)), max(0.01, 4.0-0.0477*(edad-37)); Cl1, Cl2, Cl3 = max(0.001, 2.77*((peso/74.5)**0.336)-0.0149*(edad-37)), max(0.001, 1.94-0.0280*(edad-37)), 0.197
        elif 'Eleveld' in modelo_pk:
            SIZE = ffm / calcular_ffm(70.0, 170.0, 'Masculino', 35.0)
            KMAT = ((peso**2.0)/(peso**2.0 + 2.88**2.0)) / ((70.0**2.0)/(70.0**2.0 + 2.88**2.0))
            KSEX = 1.0 if sexo == 'Masculino' else 1.0 + 0.470 * ((edad**6.0)/(edad**6.0 + 12.0**6.0)) * (1.0 - ((edad**6.0)/(edad**6.0 + 45.0**6.0)))
            V1, V2, V3 = max(0.01, 5.81*SIZE*np.exp(-0.00554*(edad-35.0))), max(0.01, 8.82*SIZE*np.exp(-0.00327*(edad-35.0))*KSEX), max(0.01, 5.03*SIZE*np.exp(-0.0315*(edad-35.0))*np.exp(-0.0260*(peso-70.0)))
            Cl1, Cl2, Cl3 = max(0.001, 2.58*(SIZE**0.75)*KMAT*KSEX*np.exp(-0.00327*(edad-35.0))), max(0.001, 1.72*(((max(0.01, 8.82*SIZE*np.exp(-0.00327*(edad-35.0))*KSEX))/8.82)**0.75)*np.exp(-0.00554*(edad-35.0))*KSEX), max(0.001, 0.124*(((max(0.01, 5.03*SIZE*np.exp(-0.0315*(edad-35.0))*np.exp(-0.0260*(peso-70.0))))/5.03)**0.75)*np.exp(-0.00554*(edad-35.0)))
        elif 'Egan' in modelo_pk: V1, V2, V3 = 7.6, 9.4, 4.7; Cl1, Cl2, Cl3 = 2.92, 1.95, 0.10
        elif 'Rigby' in modelo_pk: V1, V2, V3 = 0.963*(peso/10.5), 1.480*(peso/10.5), 0.0; Cl1, Cl2, Cl3 = 0.716*((peso/10.5)**0.75), 0.840*((peso/10.5)**0.75), 0.0
        elif 'Staschen' in modelo_pk: V1, V2, V3 = 1.44*((peso/14.6)**0.81), 3.02*((peso/14.6)**0.74), 0.0; Cl1, Cl2, Cl3 = 1.09*((peso/14.6)**(1.32*(peso**-0.20))), 0.63*((peso/14.6)**0.70), 0.0
        else: V1, V2, V3 = 5.1, 9.82, 5.42; Cl1, Cl2, Cl3 = 2.6, 2.05, 0.076

        k10, k12, k21, k13, k31 = Cl1/V1, Cl2/V1, Cl2/V2, Cl3/V1, Cl3/V3 if V3>0 else 0.0
        if 'Pediátrico' in modelo_pk: ke0 = 0.0
        elif 'Abad' in ke0_tpeak_str: ke0 = 0.120
        elif 'Egan' in ke0_tpeak_str: ke0 = 0.433
        elif 'Eleveld' in ke0_tpeak_str: ke0 = max(0.01, 1.09 * np.exp(-0.0289 * (edad - 35.0)))
        else: ke0 = max(0.01, 0.595 - 0.007 * (edad - 40))

    return k10, k12, k21, k13, k31, ke0, V1, V2, V3

def clean_arr(arr):
    return [float(x) if not np.isnan(x) else None for x in arr]

# 4. ENDPOINT ÚNICO UNIFICADO
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
        else:
            apnea = Ce >= 1.5

    # === MODELOS FARMACODINÁMICOS (PD) ===
    pd_arrays = {}
    if 'Pediátrico' not in datos.modelo_pk:
        if datos.farmaco == 'Fentanilo':
            ic50_scott = max(0.1, 11.4 - 0.0675 * datos.edad_anos)
            pd_arrays['sef_scott_1985'] = FarmacoMatematica.hill(Ce, 6.9, 4.9, 19.2, 5.1).tolist()
            pd_arrays['sef_scott_1987'] = FarmacoMatematica.hill(Ce, ic50_scott, 4.3, 18.9, 5.9).tolist()
            pd_arrays['sef_scott_1991'] = FarmacoMatematica.hill(Ce, 8.1, 6.2, 25.0, 8.2).tolist()
            pd_arrays['prob_analgesia_bae'] = FarmacoMatematica.hill(Ce, 0.63, 2.24, 0.0, 100.0).tolist()
            pd_arrays['poder_theta_balanza'] = ((Ce - 5.5) / 0.55).tolist()
            pd_arrays['mvi_balanza'] = np.clip(-1.62 * ((Ce - 5.5) / 0.55) + 83.8, 0, 100).tolist()
            pd_arrays['prob_conciencia_balanza'] = (100.0 / (1.0 + np.exp(-(2.6 - 0.1508 * ((Ce - 5.5) / 0.55))))).tolist()
            pd_arrays['vol_minuto_mildh'] = (9.9 * (1.0 - (Ce / (5.49 + Ce)))).tolist()
            pd_arrays['frec_resp_mildh'] = (15.1 * (1.0 - (Ce / (3.15 + Ce)))).tolist()
            pd_arrays['paco2_mildh'] = (40.503 + (3.915 * Ce)).tolist()
        else:
            ec50_minto = max(0.1, 13.1 - 0.148 * (datos.edad_anos - 40))
            pd_arrays['sef_minto_1997'] = FarmacoMatematica.hill(Ce, ec50_minto, 2.44, 20.0, 5.5).tolist()
            pd_arrays['sef_egan_1996'] = FarmacoMatematica.hill(Ce, 19.9, 4.3, 19.0, 5.2).tolist()
            pd_arrays['sef_eleveld_2017'] = FarmacoMatematica.hill(Ce, 12.7, 2.87, 19.9, 5.66).tolist()
            pd_arrays['analgesia_abad_2022'] = FarmacoMatematica.hill(Ce_abad, 2.8, 1.9, 0.0, 100.0).tolist()

    # === INTERACCIONES PD (ISOBOLAS Y 3D) ===
    isobolas = {}
    superficie_3d = {}

    if 'Pediátrico' not in datos.modelo_pk:
        max_y = max(np.max(Cp), np.max(Ce)) * 1.1 if max(np.max(Cp), np.max(Ce)) > 0 else 10.0
        ce_y = np.linspace(0, max_y, 100)
        isobolas['ce_farmaco'] = ce_y.tolist()
        
        if datos.farmaco == 'Fentanilo':
            b_50 = max(1.0, 4.9 - 0.09*(datos.edad_anos - 20)); b_95 = b_50*(5.4/3.3); fr_sm = (0.50*ce_y)/(0.75+ce_y)
            isobolas['propofol_smith_loc'] = {'ce_50': clean_arr(b_50*(1-fr_sm)), 'ce_95': clean_arr(b_95*(1-fr_sm))}
            fr_som = (0.95*ce_y**1.4)/(0.63**1.4+ce_y**1.4)
            isobolas['propofol_smith_somatica'] = {'ce_50': clean_arr(15.2*(1-fr_som)), 'ce_95': clean_arr(27.4*(1-fr_som))}
            
            for k_eff in ['Disminución PAS 15%', 'Disminución PAS 30%', 'Disminución PAS 40%', 'Disminución FC 15%', 'Disminución FC 30%', 'Respuesta Somática', 'Supresión Aumento PAS 15%']:
                isobolas[f"propofol_kazama_{k_eff.replace(' ', '').replace('%', '').lower()}"] = {'ce_50': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.50, k_eff)), 'ce_95': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.95, k_eff))}
            # Mapeo manual para No Resp Somática a Respuesta Somática
            isobolas["propofol_kazama_no_respuestasomatica"] = {'ce_50': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.50, 'Respuesta Somática')), 'ce_95': clean_arr(FarmacoMatematica.kazama_1998_iso_cprop(ce_y, 0.95, 'Respuesta Somática'))}
            
            for k_eff in ['LOC', 'Respuesta Simpática', 'Respuesta Somática']:
                isobolas[f"sevo_katoh_{k_eff.replace(' ', '').lower()}"] = {'ce_50': clean_arr(FarmacoMatematica.katoh_1999_iso_csevo(ce_y, 0.50, k_eff)), 'ce_95': clean_arr(FarmacoMatematica.katoh_1999_iso_csevo(ce_y, 0.95, k_eff))}
            
            isobolas['sevo_vereecke_somatica'] = {'ce_50': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 1.73, 2.07, 0.931, 6.40, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 1.73, 2.07, 0.931, 6.40, 0.95))}
            isobolas['sevo_vereecke_simpatica'] = {'ce_50': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 4.60, 0.43, 0.931, 6.40, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.vereecke_iso_cs(ce_y, 4.60, 0.43, 0.931, 6.40, 0.95))}
            isobolas['propofol_wang_lma'] = {'ce_50': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 9.77, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 9.77, 0.95))}
            isobolas['sevo_wang_lma'] = {'ce_50': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 4.29, 0.50)), 'ce_95': clean_arr(FarmacoMatematica.wang_2026_iso_cm(ce_y, 4.29, 0.95))}
        
        else: # Remifentanilo Isobolas
            for e in ['Laringoscopia', 'LOC']:
                isobolas[f"propofol_bouillon_{e.lower()[:3]}"] = {'ce_50': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.50, e)), 'ce_95': clean_arr(FarmacoMatematica.bouillon_2004_iso_cprop(ce_y, 0.95, e))}
            isobolas["propofol_bouillon_bis"] = {'ce_50': clean_arr(FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_y, 60.0)), 'ce_95': clean_arr(FarmacoMatematica.bouillon_2004_bis_iso_cprop(ce_y, 40.0))}

            p_greco = {'propofol_kern_lar': (5.60, 48.9, 33.2, 2.2, False), 'propofol_kern_mec': (4.16, 8.84, 8.2, 8.3, False), 'propofol_kern_elec': (4.56, 21.3, 14.7, 6.0, False), 'propofol_johnson_moaa_1': (2.2, 33.1, 3.6, 5.0, False), 'propofol_johnson_moaa_2': (1.3, 10.5, 2.8, 3.5, True), 'propofol_kern_moaa_3': (1.8, 12.5, 5.1, 5.8, False), 'propofol_kern_moaa_4': (1.8, 12.5, 5.1, 5.8, True), 'sevo_johnson_moaa_1': (0.74, 50.9, 9.4, 5.2, False), 'sevo_johnson_moaa_2': (0.74, 50.9, 9.4, 5.2, True), 'sevo_johnson_algo_30': (0.83, 1.3, 0.9, 2.7, False)}
            for k, v in p_greco.items(): isobolas[k] = {'ce_50': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.50, v[4])), 'ce_95': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.95, v[4]))}
            isobolas["sevo_manyam_bis"] = {'ce_50': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, 2.37, 38.02, 0.52, 1.12, 0.60, True)), 'ce_95': clean_arr(FarmacoMatematica.greco_iso_cs(ce_y, 2.37, 38.02, 0.52, 1.12, 0.40, True))}

            p_man = {'sevo_manyam_moaa_1': (7.30, 7.84, 0.23, 3.94), 'sevo_manyam_moaa_4': (4.19, 4.25, 0.28, 0.58), 'sevo_manyam_mec': (3.82, 2.43, 0.54, 1.27), 'sevo_manyam_term': (3.38, 1.32, 0.55, 3.47), 'sevo_manyam_elec': (3.27, 0.97, 0.088, 1.09), 'sevo_manyam_lar': (3.70, 2.36, 0.54, 1.22)}
            for k, v in p_man.items(): isobolas[k] = {'ce_50': clean_arr(FarmacoMatematica.manyam_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.50)), 'ce_95': clean_arr(FarmacoMatematica.manyam_iso_cs(ce_y, v[0], v[1], v[2], v[3], 0.95))}

        # NUEVO: Construcción de la matriz 3D en el servidor
        if datos.modelo_3d != 'Ninguna' and datos.modelo_3d in isobolas:
            is_prop = 'propofol' in datos.modelo_3d
            if datos.farmaco == 'Fentanilo':
                y_max, x_max = 10.0, 20.0 if is_prop else 8.0
                ylab = 'Ce Fentanilo (ng/ml)'
            else:
                y_max, x_max = 15.0, 15.0 if is_prop else 8.0
                ylab = 'Ce Remifentanilo (ng/ml)'
                
            xlab = 'Ce Propofol (ug/ml)' if is_prop else 'etSEV (%)'
            zlab = 'Valor BIS' if 'bis' in datos.modelo_3d else 'Probabilidad (%)'
            
            res = 25
            x_m = np.linspace(0, x_max, res)
            y_m = np.linspace(0, y_max, res)
            X, Y = np.meshgrid(x_m, y_m)
            
            Z = np.zeros_like(X)
            
            if datos.farmaco == 'Fentanilo':
                if 'smith_loc' in datos.modelo_3d:
                    b_50 = max(1.0, 4.9 - 0.09*(datos.edad_anos - 20))
                    fr = (0.50*Y)/(0.75+Y)
                    u50 = np.clip(b_50 * (1.0 - fr), 1e-5, None)
                    ratio = np.clip((b_50*(5.4/3.3) * (1.0 - fr)) / u50, 1.01, None)
                    Z = (X**(np.log(19.0)/np.log(ratio))) / (X**(np.log(19.0)/np.log(ratio)) + u50**(np.log(19.0)/np.log(ratio))) * 100.0
                elif 'smith_somatica' in datos.modelo_3d or 'no_respuestasomatica' in datos.modelo_3d:
                    fr = (0.95*Y**1.4)/(0.63**1.4+Y**1.4)
                    u50 = np.clip(15.2 * (1.0 - fr), 1e-5, None)
                    ratio = np.clip((27.4 * (1.0 - fr)) / u50, 1.01, None)
                    Z = (X**(np.log(19.0)/np.log(ratio))) / (X**(np.log(19.0)/np.log(ratio)) + u50**(np.log(19.0)/np.log(ratio))) * 100.0
                elif 'kazama' in datos.modelo_3d:
                    eff = datos.modelo_3d.replace('propofol_kazama_', '')
                    eff_map = {
                        'disminuciónpas15': 'Disminución PAS 15%', 'disminuciónpas30': 'Disminución PAS 30%',
                        'disminuciónpas40': 'Disminución PAS 40%', 'disminuciónfc15': 'Disminución FC 15%',
                        'disminuciónfc30': 'Disminución FC 30%', 'no_respuestasomatica': 'Respuesta Somática',
                        'supresiónaumentopas15': 'Supresión Aumento PAS 15%'
                    }
                    if eff in eff_map:
                        ec_s, ec_r, alpha, n = {
                            'Disminución PAS 15%': (3.6, 9.7, 1.5, 1.5), 'Disminución PAS 30%': (8.1, 20.5, 3.1, 1.6),
                            'Disminución PAS 40%': (17.7, 195.1, 41.5, 8.5), 'Disminución FC 15%': (14.4, 3.5, 1.65, 3.3),
                            'Disminución FC 30%': (20.5, 6.7, 1.2, 4.6), 'Respuesta Somática': (13.8, 9.7, 6.8, 2.63),
                            'Supresión Aumento PAS 15%': (27.7, 5.3, 3.7, 1.7)
                        }[eff_map[eff]]
                        Z = FarmacoMatematica.greco_prob(X, Y, ec_s, ec_r, alpha, n, False, True)
                elif 'katoh' in datos.modelo_3d:
                    eff = 'LOC' if 'loc' in datos.modelo_3d else ('Respuesta Simpática' if 'simpatica' in datos.modelo_3d else 'Respuesta Somática')
                    b50, b95, c50f, gf, mred = (0.62,0.71,7.3,1.2,1.0) if eff=='LOC' else ((4.15,6.26,0.78,1.5,0.95) if eff=='Respuesta Simpática' else (1.77,2.21,1.08,1.0,0.80))
                    fr = (mred * Y**gf) / (c50f**gf + Y**gf)
                    u50 = np.clip(b50 * (1.0 - fr), 1e-5, None)
                    ratio = np.clip((b95 * (1.0 - fr)) / u50, 1.01, None)
                    Z = (X**(np.log(19.0)/np.log(ratio))) / (X**(np.log(19.0)/np.log(ratio)) + u50**(np.log(19.0)/np.log(ratio))) * 100.0
                elif 'vereecke' in datos.modelo_3d:
                    c50f, c50s = (2.07, 1.73) if 'somatica' in datos.modelo_3d else (0.43, 4.60)
                    Z = FarmacoMatematica.vereecke_prob(X, Y, c50s, c50f, 0.931, 6.40, True)
                elif 'wang' in datos.modelo_3d:
                    c50m = 9.77 if is_prop else 4.29
                    us = np.where((X/c50m + Y/2.58) == 0, 1e-6, (X/c50m + Y/2.58))
                    x_w, y_w = (X/c50m)/us, (Y/2.58)/us
                    U50 = 10.0**((1.0-x_w)*(1.0-y_w)*(-0.06*x_w - 1.35*y_w - 0.78*x_w*y_w))
                    n_w = 1.98*x_w + 1.98*y_w + 1.44*x_w*y_w
                    Z = (((X/c50m + Y/2.58)/U50)**n_w) / (1.0 + ((X/c50m + Y/2.58)/U50)**n_w) * 100.0
                    Z = np.where((X/c50m + Y/2.58) == 0, 0, Z)
            else: # Remifentanilo 3D
                if 'bouillon_bis' in datos.modelo_3d:
                    u_bis = (np.clip(X,0,None)/4.47) + (np.clip(Y,0,None)/19.3)
                    Z = 97.4 - 97.4 * ((u_bis**1.43) / (1.0 + u_bis**1.43))
                elif 'bouillon' in datos.modelo_3d:
                    ipre = 0.60 if 'loc' in datos.modelo_3d else 1.05
                    Ys = np.clip(Y, 1e-6, None)
                    n_rem = Ys**0.97
                    ipost = ipre * (1.0 - (n_rem / (n_rem + (1.07*ipre)**0.97)))
                    Xs = np.clip(X, 1e-6, None)
                    Z = (Xs**5.1) / (Xs**5.1 + (8.04*ipost)**5.1) * 100.0
                elif 'kern' in datos.modelo_3d or 'johnson' in datos.modelo_3d:
                    m = datos.modelo_3d
                    if is_prop:
                        if 'lar' in m: ecs, ecr, alp, n, inv = 5.60, 48.9, 33.2, 2.2, False
                        elif 'mec' in m: ecs, ecr, alp, n, inv = 4.16, 8.84, 8.2, 8.3, False
                        elif 'elec' in m: ecs, ecr, alp, n, inv = 4.56, 21.3, 14.7, 6.0, False
                        elif 'moaa_1' in m: ecs, ecr, alp, n, inv = 2.2, 33.1, 3.6, 5.0, False
                        elif 'moaa_2' in m: ecs, ecr, alp, n, inv = 1.3, 10.5, 2.8, 3.5, True
                        elif 'moaa_3' in m: ecs, ecr, alp, n, inv = 1.8, 12.5, 5.1, 5.8, False
                        elif 'moaa_4' in m: ecs, ecr, alp, n, inv = 1.8, 12.5, 5.1, 5.8, True
                    else:
                        if 'moaa_1' in m: ecs, ecr, alp, n, inv = 0.74, 50.9, 9.4, 5.2, False
                        elif 'moaa_2' in m: ecs, ecr, alp, n, inv = 0.74, 50.9, 9.4, 5.2, True
                        elif 'algo' in m: ecs, ecr, alp, n, inv = 0.83, 1.3, 0.9, 2.7, False
                    Z = FarmacoMatematica.greco_prob(X, Y, ecs, ecr, alp, n, inv, True)
                elif 'manyam_bis' in datos.modelo_3d:
                    Z = FarmacoMatematica.greco_prob(X, Y, 2.37, 38.02, 0.52, 1.12, True, True)
                elif 'manyam' in datos.modelo_3d:
                    m = datos.modelo_3d
                    if 'moaa_1' in m: b0,b1,b2,b3 = 7.30, 7.84, 0.23, 3.94
                    elif 'moaa_4' in m: b0,b1,b2,b3 = 4.19, 4.25, 0.28, 0.58
                    elif 'mec' in m: b0,b1,b2,b3 = 3.82, 2.43, 0.54, 1.27
                    elif 'term' in m: b0,b1,b2,b3 = 3.38, 1.32, 0.55, 3.47
                    elif 'elec' in m: b0,b1,b2,b3 = 3.27, 0.97, 0.088, 1.09
                    elif 'lar' in m: b0,b1,b2,b3 = 3.70, 2.36, 0.54, 1.22
                    Z = FarmacoMatematica.manyam_prob(X, Y, b0, b1, b2, b3, True)

            superficie_3d = {
                "x_mesh": x_m.tolist(),
                "y_mesh": y_m.tolist(),
                "z_mesh": np.round(Z, 2).tolist(),
                "iso50_x": clean_arr(isobolas[datos.modelo_3d]['ce_50']),
                "iso50_y": clean_arr(isobolas['ce_farmaco']),
                "iso95_x": clean_arr(isobolas[datos.modelo_3d]['ce_95']),
                "iso95_y": clean_arr(isobolas['ce_farmaco']),
                "xlabel": xlab,
                "ylabel": ylab,
                "zlabel": zlab,
                "title": datos.modelo_3d
            }

    return {
        "tiempo_minutos": np.round(t_sim, 2).tolist(),
        "cp": np.round(Cp, 3).tolist(),
        "ce": np.round(Ce, 3).tolist(),
        "alertas_clinicas": {"torax_lenoso": torax_lenoso.tolist(), "depresion_respiratoria": depresion_resp.tolist(), "apnea": apnea.tolist()},
        "farmacodinamia_pd": {k: np.round(v, 2).tolist() for k, v in pd_arrays.items()},
        "isobolas_interaccion": isobolas,
        "superficie_3d": superficie_3d
    }
