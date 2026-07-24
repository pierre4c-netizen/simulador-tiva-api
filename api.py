from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

# 1. Definición de las estructuras de entrada
class EventoTIVA(BaseModel):
    tipo: str
    ini_min: float
    fin_min: float
    tasa_ug_min: float

class PeticionSimulacion(BaseModel):
    peso_kg: float
    altura_cm: float
    edad_anos: float
    sexo: str
    modelo_pk: str
    ke0_tpeak: str
    eventos: List[EventoTIVA]
    minutos_simulacion: int = 1440 # Capacidad máxima de 24h

app = FastAPI(title="Motor TIVA Analítico Avanzado Completo")

# 2. Funciones Matemáticas Puras
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
    def objective(ke):
        Cp_t = A * np.exp(-l1 * target_tpeak) + B * np.exp(-l2 * target_tpeak) + C_coeff * np.exp(-l3 * target_tpeak)
        Ce_t = ke * (A * get_term(l1, ke, target_tpeak) + B * get_term(l2, ke, target_tpeak) + C_coeff * get_term(l3, ke, target_tpeak))
        return Cp_t - Ce_t
    try: res = root_scalar(objective, bracket=[0.001, 3.0], method='brentq'); return res.root
    except Exception: return 0.147 

def get_pk_params(modelo_pk, ke0_tpeak_str, peso, altura, sexo, edad_paciente):
    if modelo_pk == 'Scott 1987 (Fijo)': V1, V2, V3 = 12.7, 50.7, 274.0; Cl1, Cl2, Cl3 = 0.574, 4.01, 1.95
    elif modelo_pk == 'Shafer 1990 (Fijo)': V1, V2, V3 = 6.09, 28.1, 228.0; Cl1, Cl2, Cl3 = 0.504, 2.87, 1.37
    elif modelo_pk == 'Shafer 1990 (Peso)': V1, V2, V3 = 0.105 * peso, 0.446 * peso, 3.37 * peso; Cl1, Cl2, Cl3 = 0.00838 * peso, 0.0474 * peso, 0.0199 * peso
    elif modelo_pk == 'Bae 2020 (Alométrico)':
        f_vol, f_cl = (peso / 70.0) ** 1.23, (peso / 70.0) ** 0.313  
        V1, V2, V3 = 10.1 * f_vol, 26.5 * f_vol, 206.0 * f_vol; Cl1, Cl2, Cl3 = 0.704 * f_cl, 2.38 * f_cl, 1.49 * f_cl
    elif modelo_pk == 'Ginsberg 1996 (Pediatría, Peso y Edad)':
        V1, V2, V3 = max(0.001, 0.43 * (peso - 19.8) + 5.8), max(0.001, 6.2 * (edad_paciente - 6.4) + 34.4), 0.0
        Cl1, Cl2, Cl3 = max(0.001, 0.01 * (peso - 19.8) + 0.35), max(0.001, 0.82), 0.0
    elif modelo_pk == 'Maharaj 2019 (Pediatría, Alométrico)':
        V1, V2, V3 = 10.8 * (peso / 70.0), 417.0 * (peso / 70.0), 0.0
        Cl1, Cl2, Cl3 = (32.5 / 60.0) * ((peso / 70.0) ** 0.75), (104.0 / 60.0) * ((peso / 70.0) ** 0.75), 0.0
    elif modelo_pk == 'Okada 2024 (Pediatría, Alométrico)':
        altura_m = altura / 100.0; bmi = peso / (altura_m ** 2) if altura_m > 0 else 20.0
        if sexo == 'Masculino': m3 = (9270.0 * peso) / (6680.0 + 216.0 * bmi); a_mat, A50, c_mat = 0.88, 13.4, 12.7
        else: m3 = (9270.0 * peso) / (8780.0 + 244.0 * bmi); a_mat, A50, c_mat = 1.11, 7.1, 1.1
        mf = a_mat + ((1.0 - a_mat) * (edad_paciente / A50)**c_mat) / (1.0 + (edad_paciente / A50)**c_mat)
        ffm = mf * m3; V1, V2, V3 = 0.024 * (ffm ** 0.64), 0.30, 11.0; Cl1, Cl2, Cl3 = 0.037 * (peso ** 0.46), 0.021, 0.066
    else: V1, V2, V3 = 12.7, 50.7, 274.0; Cl1, Cl2, Cl3 = 0.574, 4.01, 1.95 # Fallback
    
    k10 = Cl1/V1 if V1>0 else 0; k12 = Cl2/V1 if V1>0 else 0; k21 = Cl2/V2 if V2>0 else 0; k13 = Cl3/V1 if V1>0 else 0; k31 = Cl3/V3 if V3>0 else 0
    if 'Pediatría' in modelo_pk: ke0 = 0.0
    else:
        if ke0_tpeak_str == 'Ke0 0.147': ke0 = 0.147
        elif ke0_tpeak_str == 'Ke0 0.108': ke0 = 0.108
        elif ke0_tpeak_str == 'Ke0 0.105': ke0 = 0.105
        elif ke0_tpeak_str == 'Tpeak 4 min': ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 4.0)
        else: ke0 = 0.147
    return k10, k12, k21, k13, k31, ke0, V1, V2, V3

# 3. Endpoint Principal de Simulación Total
@app.post("/simular/fentanilo")
def calcular_fentanilo_completo(datos: PeticionSimulacion) -> Dict[str, Any]:
    
    k10, k12, k21, k13, k31, ke0, V1, V2, V3 = get_pk_params(
        datos.modelo_pk, datos.ke0_tpeak, datos.peso_kg, 
        datos.altura_cm, datos.sexo, datos.edad_anos
    )

    def ode_sys(y, t_ode):
        x1, x2, x3, ce = y
        entrada_total = sum([ev.tasa_ug_min for ev in datos.eventos if ev.ini_min <= t_ode <= ev.fin_min])
        dx1 = entrada_total - (k10 + k12 + k13)*x1 + k21*x2 + k31*x3
        return [dx1, k12*x1 - k21*x2, k13*x1 - k31*x3, ke0 * ((x1 / V1) - ce)]

    # Determinar el tiempo de simulación dinámico (Resolución: 10 puntos por minuto para graficado perfecto)
    max_evento_min = max([e.fin_min for e in datos.eventos]) if datos.eventos else 60
    minutos_totales = max(datos.minutos_simulacion, int(max_evento_min) + 60)
    puntos = (minutos_totales * 10) + 1
    t_sim = np.linspace(0, float(minutos_totales), puntos)
    
    solucion = odeint(ode_sys, [0.0, 0.0, 0.0, 0.0], t_sim)
    Cp = solucion[:, 0] / V1
    Ce = solucion[:, 3]

    max_cp_idx = np.argmax(Cp); max_ce_idx = np.argmax(Ce)
    max_cp = float(Cp[max_cp_idx]); max_ce = float(Ce[max_ce_idx])

    # === ALERTAS Y VENTANAS CLÍNICAS ===
    torax_lenoso = np.zeros_like(t_sim, dtype=bool)
    depresion_resp = np.zeros_like(t_sim, dtype=bool)
    urpa = np.zeros_like(t_sim, dtype=bool)
    cam = np.zeros_like(t_sim, dtype=bool)
    agb = np.zeros_like(t_sim, dtype=bool)
    iet = np.zeros_like(t_sim, dtype=bool)

    if 'Pediatría' not in datos.modelo_pk:
        # Tórax Leñoso
        dce_dt = ke0 * (Cp - Ce)
        estado_tl_activo = False
        umbral_gradiente = ke0 * 21.5 
        for i in range(len(t_sim)):
            if not estado_tl_activo:
                if dce_dt[i] >= umbral_gradiente and Cp[i] >= 21.5: estado_tl_activo = True
            else:
                if Cp[i] <= 6.9: estado_tl_activo = False
            torax_lenoso[i] = estado_tl_activo
            
        # Depresión respiratoria y ventanas
        depresion_resp = Ce >= 1.0
        urpa = (Ce >= 0.6) & (Ce <= 1.0)
        cam = (Ce >= 0.9) & (Ce <= 1.1)
        agb = (Ce >= 1.0) & (Ce <= 2.0)
        iet = (Ce >= 2.0) & (Ce <= 3.0)

    # === MODELOS FARMACODINÁMICOS (PD) ===
    pd_arrays = {}
    if 'Pediatría' not in datos.modelo_pk:
        # SEF
        ic50_scott_87 = max(0.1, 11.4 - 0.0675 * datos.edad_anos)
        pd_arrays['sef_scott_1985'] = 19.2 - 14.1 * (Ce**4.9) / (6.9**4.9 + Ce**4.9)
        pd_arrays['sef_scott_1987'] = 18.9 - 13.0 * (Ce**4.3) / (ic50_scott_87**4.3 + Ce**4.3)
        pd_arrays['sef_scott_1991'] = 25.0 - 16.8 * (Ce**6.2) / (8.1**6.2 + Ce**6.2)
        
        # Bae & Balanza
        pd_arrays['prob_analgesia_bae'] = 100.0 * (Ce**2.24) / (0.63**2.24 + Ce**2.24)
        pd_arrays['poder_theta_balanza'] = (Ce - 5.5) / 0.55
        pd_arrays['mvi_balanza'] = np.clip(-1.62 * pd_arrays['poder_theta_balanza'] + 83.8, 0, 100)
        pd_arrays['prob_conciencia_balanza'] = 100.0 / (1.0 + np.exp(-(2.6 - 0.1508 * pd_arrays['poder_theta_balanza'])))
        
        # Mildh
        pd_arrays['vol_minuto_mildh'] = 9.9 * (1.0 - (Ce / (5.49 + Ce)))
        pd_arrays['frec_resp_mildh'] = 15.1 * (1.0 - (Ce / (3.15 + Ce)))
        pd_arrays['paco2_mildh'] = 40.503 + (3.915 * Ce)

    # === INTERACCIONES PD (ISOBOLAS) ===
    # Calculamos las curvas del espacio de diseño (no dependientes del tiempo, sino de concentraciones meta)
    isobolas = {}
    if 'Pediatría' not in datos.modelo_pk:
        max_y = max(max_cp, max_ce) * 1.1 if max(max_cp, max_ce) > 0 else 5.0
        ce_y = np.linspace(0, max_y, 100)
        
        # Propofol
        base_50 = max(1.0, 4.9 - 0.09 * (datos.edad_anos - 20)); base_95 = base_50 * (5.4 / 3.3)
        frac_red_conciencia = (0.50 * ce_y) / (0.75 + ce_y)
        isobolas['propofol_conciencia'] = {
            'ce_fentanilo': ce_y.tolist(),
            'ce_propofol_50': (base_50 * (1.0 - frac_red_conciencia)).tolist(),
            'ce_propofol_95': (base_95 * (1.0 - frac_red_conciencia)).tolist()
        }

        K_50 = 1.0; K_95 = (0.95 / 0.05)**(1/3.7); Cp50_prop_base = 18.6; Cp50_fent_base = 9.1; alpha = 2.9
        denom_hemo = (1.0 / Cp50_prop_base) + (alpha * ce_y) / (Cp50_prop_base * Cp50_fent_base)
        x50_hemo = (K_50 - (ce_y / Cp50_fent_base)) / denom_hemo
        x95_hemo = (K_95 - (ce_y / Cp50_fent_base)) / denom_hemo
        x50_hemo[x50_hemo < 0] = np.nan; x95_hemo[x95_hemo < 0] = np.nan
        isobolas['propofol_simpatica'] = {
            'ce_fentanilo': ce_y.tolist(),
            'ce_propofol_50': np.where(np.isnan(x50_hemo), None, x50_hemo).tolist(),
            'ce_propofol_95': np.where(np.isnan(x95_hemo), None, x95_hemo).tolist()
        }

        frac_red_som_prop = (0.95 * ce_y**1.4) / (0.63**1.4 + ce_y**1.4)
        isobolas['propofol_somatica'] = {
            'ce_fentanilo': ce_y.tolist(),
            'ce_propofol_50': (15.2 * (1.0 - frac_red_som_prop)).tolist(),
            'ce_propofol_95': (27.4 * (1.0 - frac_red_som_prop)).tolist()
        }

        # Sevoflurano
        frac_red_awa = (1.0 * ce_y**1.2) / (7.3**1.2 + ce_y**1.2)
        isobolas['sevo_conciencia'] = {
            'ce_fentanilo': ce_y.tolist(),
            'et_sevo_50': (0.62 * (1.0 - frac_red_awa)).tolist(),
            'et_sevo_95': (0.71 * (1.0 - frac_red_awa)).tolist()
        }

        frac_red_bar = (0.95 * ce_y**1.5) / (0.78**1.5 + ce_y**1.5)
        isobolas['sevo_simpatica'] = {
            'ce_fentanilo': ce_y.tolist(),
            'et_sevo_50': (4.15 * (1.0 - frac_red_bar)).tolist(),
            'et_sevo_95': (6.26 * (1.0 - frac_red_bar)).tolist()
        }

        frac_red_mac = (0.80 * ce_y) / (1.08 + ce_y)
        isobolas['sevo_somatica'] = {
            'ce_fentanilo': ce_y.tolist(),
            'et_sevo_50': (1.77 * (1.0 - frac_red_mac)).tolist(),
            'et_sevo_95': (2.21 * (1.0 - frac_red_mac)).tolist()
        }

    # === EMPAQUETADO DE RESPUESTA JSON ===
    # Convertimos los arreglos a listas con precisión redondeada para reducir peso de red
    return {
        "estado": "Exito",
        "parametros_calculados": {
            "ke0_aplicado": round(ke0, 5),
            "v1": round(V1, 2), "v2": round(V2, 2), "v3": round(V3, 2),
            "max_cp": round(max_cp, 3), "max_ce": round(max_ce, 3)
        },
        "tiempo_minutos": np.round(t_sim, 2).tolist(),
        "cp": np.round(Cp, 3).tolist(),
        "ce": np.round(Ce, 3).tolist(),
        
        "alertas_clinicas": {
            "torax_lenoso": torax_lenoso.tolist(),
            "depresion_respiratoria": depresion_resp.tolist()
        },
        "ventanas_terapeuticas": {
            "urpa": urpa.tolist(),
            "cam": cam.tolist(),
            "agb": agb.tolist(),
            "iet": iet.tolist()
        },
        "farmacodinamia_pd": {k: np.round(v, 2).tolist() for k, v in pd_arrays.items()},
        "isobolas_interaccion": isobolas
    }
