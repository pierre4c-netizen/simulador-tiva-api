from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

# 1. Definición de las estructuras de entrada
class EventoTIVA(BaseModel):
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
    minutos_simulacion: int = 120 # Por defecto simulamos 2 horas

app = FastAPI(title="Motor TIVA Analítico Avanzado")

# 2. Funciones Matemáticas Puras (Extraídas de tu Jupyter)
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
    
    def get_term(l, ke, t): 
        return t * np.exp(-ke * t) if abs(ke - l) < 1e-6 else (np.exp(-l * t) - np.exp(-ke * t)) / (ke - l)
        
    def objective(ke):
        Cp_t = A * np.exp(-l1 * target_tpeak) + B * np.exp(-l2 * target_tpeak) + C_coeff * np.exp(-l3 * target_tpeak)
        Ce_t = ke * (A * get_term(l1, ke, target_tpeak) + B * get_term(l2, ke, target_tpeak) + C_coeff * get_term(l3, ke, target_tpeak))
        return Cp_t - Ce_t
        
    try: 
        res = root_scalar(objective, bracket=[0.001, 3.0], method='brentq')
        return res.root
    except Exception: 
        return 0.147 

def get_pk_params(modelo_pk, ke0_tpeak_str, peso, altura, sexo, edad_paciente):
    if modelo_pk == 'Scott 1987 (Fijo)': 
        V1, V2, V3 = 12.7, 50.7, 274.0; Cl1, Cl2, Cl3 = 0.574, 4.01, 1.95
    elif modelo_pk == 'Shafer 1990 (Fijo)': 
        V1, V2, V3 = 6.09, 28.1, 228.0; Cl1, Cl2, Cl3 = 0.504, 2.87, 1.37
    elif modelo_pk == 'Shafer 1990 (Peso)': 
        V1, V2, V3 = 0.105 * peso, 0.446 * peso, 3.37 * peso; Cl1, Cl2, Cl3 = 0.00838 * peso, 0.0474 * peso, 0.0199 * peso
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
        if sexo == 'Masculino': 
            m3 = (9270.0 * peso) / (6680.0 + 216.0 * bmi); a_mat, A50, c_mat = 0.88, 13.4, 12.7
        else: 
            m3 = (9270.0 * peso) / (8780.0 + 244.0 * bmi); a_mat, A50, c_mat = 1.11, 7.1, 1.1
        mf = a_mat + ((1.0 - a_mat) * (edad_paciente / A50)**c_mat) / (1.0 + (edad_paciente / A50)**c_mat)
        ffm = mf * m3
        V1, V2, V3 = 0.024 * (ffm ** 0.64), 0.30, 11.0; Cl1, Cl2, Cl3 = 0.037 * (peso ** 0.46), 0.021, 0.066
    
    k10 = Cl1/V1 if V1>0 else 0; k12 = Cl2/V1 if V1>0 else 0; k21 = Cl2/V2 if V2>0 else 0; k13 = Cl3/V1 if V1>0 else 0; k31 = Cl3/V3 if V3>0 else 0
    
    if 'Pediatría' in modelo_pk: 
        ke0 = 0.0
    else:
        if ke0_tpeak_str == 'Ke0 0.147': ke0 = 0.147
        elif ke0_tpeak_str == 'Ke0 0.108': ke0 = 0.108
        elif ke0_tpeak_str == 'Ke0 0.105': ke0 = 0.105
        elif ke0_tpeak_str == 'Tpeak 4 min': ke0 = calcular_ke0_para_tpeak(k10, k12, k21, k13, k31, 4.0)
        else: ke0 = 0.147
        
    return k10, k12, k21, k13, k31, ke0, V1, V2, V3


# 3. El Endpoint Principal
@app.post("/simular/fentanilo")
def calcular_fentanilo(datos: PeticionSimulacion) -> Dict[str, Any]:
    
    k10, k12, k21, k13, k31, ke0, V1, V2, V3 = get_pk_params(
        datos.modelo_pk, datos.ke0_tpeak, datos.peso_kg, 
        datos.altura_cm, datos.sexo, datos.edad_anos
    )

    def ode_sys(y, t_ode):
        x1, x2, x3, ce = y
        entrada_total = sum([ev.tasa_ug_min for ev in datos.eventos if ev.ini_min <= t_ode <= ev.fin_min])
        dx1 = entrada_total - (k10 + k12 + k13)*x1 + k21*x2 + k31*x3
        dx2 = k12*x1 - k21*x2
        dx3 = k13*x1 - k31*x3
        dce = ke0 * ((x1 / V1) - ce)
        return [dx1, dx2, dx3, dce]

    # Resolución matemática (10 puntos por minuto para precisión clínica)
    puntos_totales = (datos.minutos_simulacion * 10) + 1
    t_sim = np.linspace(0, float(datos.minutos_simulacion), puntos_totales)
    
    solucion = odeint(ode_sys, [0.0, 0.0, 0.0, 0.0], t_sim)
    Cp = solucion[:, 0] / V1
    Ce = solucion[:, 3]

    # Lógica de Tórax Leñoso
    torax_lenoso = np.zeros_like(t_sim, dtype=bool)
    if 'Pediatría' not in datos.modelo_pk:
        dce_dt = ke0 * (Cp - Ce)
        estado_tl_activo = False
        umbral_gradiente = ke0 * 21.5 
        for i in range(len(t_sim)):
            if not estado_tl_activo:
                if dce_dt[i] >= umbral_gradiente and Cp[i] >= 21.5: 
                    estado_tl_activo = True
            else:
                if Cp[i] <= 6.9: 
                    estado_tl_activo = False
            torax_lenoso[i] = estado_tl_activo

    # Modelos Farmacodinámicos (PD)
    pd_results = {}
    if 'Pediatría' not in datos.modelo_pk:
        ic50_scott = max(0.1, 11.4 - 0.0675 * datos.edad_anos)
        pd_results['sef_scott_1987'] = (18.9 - 13.0 * (Ce**4.3) / (ic50_scott**4.3 + Ce**4.3)).tolist()
        pd_results['prob_analgesia_bae'] = (100.0 * (Ce**2.24) / (0.63**2.24 + Ce**2.24)).tolist()
        pd_results['vol_minuto_mildh'] = (9.9 * (1.0 - (Ce / (5.49 + Ce)))).tolist()
        pd_results['poder_theta_balanza'] = ((Ce - 5.5) / 0.55).tolist()

    return {
        "estado": "Exito",
        "ke0_aplicado": round(ke0, 5),
        "tiempo_minutos": np.round(t_sim, 2).tolist(),
        "concentracion_plasmatica": np.round(Cp, 3).tolist(),
        "concentracion_efecto": np.round(Ce, 3).tolist(),
        "alertas": {
            "torax_lenoso": torax_lenoso.tolist(),
        },
        "farmacodinamia": pd_results
    }