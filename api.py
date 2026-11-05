from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from scipy.integrate import odeint

# IMPORTANTE: Importamos toda la matemática y lógicas del nuevo archivo
from farmaco_math import FarmacoMatematica, get_pk_params, clean_arr

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
# 2. ENDPOINT ÚNICO UNIFICADO
# ==========================================
@app.post("/simular")
def calcular_simulacion_completa(datos: PeticionSimulacion) -> Dict[str, Any]:
    # Obtener parámetros PK usando el import
    k10, k12, k21, k13, k31, ke0, V1, V2, V3 = get_pk_params(
        datos.farmaco, datos.modelo_pk, datos.ke0_tpeak, datos.peso_kg, 
        datos.altura_cm, datos.sexo, datos.edad_anos
    )

    def ode_sys(y, t_ode):
        x1, x2, x3, ce = y
        entrada_total = sum([ev.tasa_ug_min for ev in datos.eventos if ev.ini_min <= t_ode < ev.fin_min or (t_ode == ev.fin_min and ev.fin_min == 1440.0)])
        dx1 = entrada_total - (k10 + k12 + k13)*x1 + k21*x2 + k31*x3
        return [dx1, k12*x1 - k21*x2, k13*x1 - k31*x3, ke0 * ((x1 / V1) - ce)]

    max_evento_min = max([e.fin_min for e in datos.eventos]) if datos.eventos else 60
    minutos_totales = max(datos.minutos_simulacion, int(max_evento_min) + 60)
    t_sim = np.linspace(0, float(minutos_totales), (minutos_totales * 10) + 1)
    
    puntos_criticos = sorted(list({t for ev in datos.eventos for t in (ev.ini_min, ev.fin_min) if 0.0 <= t <= minutos_totales}))
    sol = odeint(ode_sys, [0.0, 0.0, 0.0, 0.0], t_sim, tcrit=puntos_criticos if len(puntos_criticos) > 0 else None)
    Cp = np.clip(sol[:, 0] / V1, 0.0, None)
    Ce = np.clip(sol[:, 3], 0.0, None)

    # === ALERTAS CLÍNICAS ===
    torax_lenoso, depresion_resp, apnea = np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool), np.zeros_like(t_sim, dtype=bool)
    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina' or 'Propósito General' in datos.modelo_pk:
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
    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina' or 'Propósito General' in datos.modelo_pk:
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
            pd_arrays['Intensidad Disociativa (Kamp 2021)'] = FarmacoMatematica.ketamina_pd(Ce, 'Intensidad Disociativa (Kamp 2021)').tolist()
            pd_arrays['Factor de Tolerancia al Estimulo Mecanico (Kamp 2021)'] = FarmacoMatematica.ketamina_pd(Ce, 'Factor de Tolerancia al Estimulo Mecanico (Kamp 2021)').tolist()
            pd_arrays['Nivel de Sedación CHWSS (Herd 2008)'] = FarmacoMatematica.herd_2008_sedacion(Ce).tolist()
            pd_arrays['Probabilidad de llegar al despertar (Herd 2008)'] = FarmacoMatematica.herd_2008_prob_despertar(Ce).tolist()
        else:
            pd_arrays['Frecuencia Borde Espectral (Minto 1997)'] = FarmacoMatematica.hill(Ce, max(0.1, 13.1 - 0.148 * (datos.edad_anos - 40)), 2.44, 20.0, 5.5).tolist()
            pd_arrays['Frecuencia Borde Espectral (Egan 1996)'] = FarmacoMatematica.hill(Ce, 19.9, 4.3, 19.0, 5.2).tolist()
            pd_arrays['Frecuencia Borde Espectral (Eleveld 2017)'] = FarmacoMatematica.hill(Ce, 12.7, 2.87, 19.9, 5.66).tolist()
            pd_arrays['Factor Tolerancia al Estimulo Mecanico (Abad 2022)'] = FarmacoMatematica.hill(Ce, 2.8, 1.9, 1.0, 3.88).tolist()

    # === INTERACCIONES PD (ISOBOLAS Y 3D) ===
    isobolas_clasicas = {}
    superficie_3d = {}
    grafico_dinamico_iso = {"activo": False}

    if 'Pediátrico' not in datos.modelo_pk or datos.farmaco == 'Ketamina' or 'Propósito General' in datos.modelo_pk:
        base_max = max(np.max(Cp), np.max(Ce)) * 1.1 if max(np.max(Cp), np.max(Ce)) > 0 else 10.0
        if datos.farmaco == 'Ketamina': max_y = max(base_max, 1000.0)
        else: max_y = max(base_max, 15.0)
        
        # 1. MODO CLÁSICO (100 puntos en Y fijos para todos los modelos del fármaco)
        ce_y_clasico = np.linspace(0, max_y, 100)
        isobolas_clasicas['ce_farmaco'] = ce_y_clasico.tolist()
        
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
                isobolas_tmp['Sevoflurano RE post-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_re, c50_s_rep, c50_r_re, gam_re), 'unit': '%'}
                isobolas_tmp['Sevoflurano RE pre-Laringoscopia (Heyse 2014)'] = {'ce_50': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 60.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'ce_95': FarmacoMatematica.heyse_2014_iso_csevo(c_y, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'base_50': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 60.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'base_95': FarmacoMatematica.heyse_2014_iso_csevo(0.0, 40.0, e0_re, c50_s_reo, c50_r_re, gam_re), 'unit': '%'}

            elif datos.farmaco == 'Ketamina':
                isobolas_tmp['Remifentanilo Probabilidad de Conservar la Ventilación Minuto (Jonkman 2018)'] = {'ce_50': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.50, solve_for='remi'), 'ce_95': FarmacoMatematica.jonkman_2018_iso_cx(c_y, 0.80, solve_for='remi'), 'base_50': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.50, 'remi'), 'base_95': FarmacoMatematica.jonkman_2018_iso_cx(0.0, 0.80, 'remi'), 'unit': 'ng/ml'}
                isobolas_tmp['Propofol LOC (Hui 1995)'] = {'ce_50': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.50, efecto='LOC', solve_for='prop'), 'ce_95': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.95, efecto='LOC', solve_for='prop'), 'base_50': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.50, efecto='LOC', solve_for='prop'), 'base_95': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.95, efecto='LOC', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol Tolerancia al Estímulo Eléctrico (Hui 1995)'] = {'ce_50': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.50, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'ce_95': FarmacoMatematica.hui_1995_iso_cx(c_y, 0.95, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'base_50': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.50, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'base_95': FarmacoMatematica.hui_1995_iso_cx(0.0, 0.95, efecto='Tolerancia al Estímulo Eléctrico', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Sevoflurano No Movimiento (Hamp 2018)'] = {'ce_50': FarmacoMatematica.hamp_2018_iso_csevo(c_y, 0.50), 'ce_95': np.full_like(c_y, np.nan), 'base_50': 2.2, 'base_95': np.nan, 'unit': '%'}
                # Nuevos Ketofol (Coulter y Shen)
                isobolas_tmp['Propofol CHWSS ≤ 2 (Coulter 2014)'] = {'ce_50': FarmacoMatematica.coulter_2014_iso_cx(c_y, 0.50, efecto='CHWSS', solve_for='prop'), 'ce_95': FarmacoMatematica.coulter_2014_iso_cx(c_y, 0.95, efecto='CHWSS', solve_for='prop'), 'base_50': FarmacoMatematica.coulter_2014_iso_cx(0.0, 0.50, efecto='CHWSS', solve_for='prop'), 'base_95': FarmacoMatematica.coulter_2014_iso_cx(0.0, 0.95, efecto='CHWSS', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol Tolerancia al Estimulo Electrico (Coulter 2014)'] = {'ce_50': FarmacoMatematica.coulter_2014_iso_cx(c_y, 0.50, efecto='Electrico', solve_for='prop'), 'ce_95': FarmacoMatematica.coulter_2014_iso_cx(c_y, 0.95, efecto='Electrico', solve_for='prop'), 'base_50': FarmacoMatematica.coulter_2014_iso_cx(0.0, 0.50, efecto='Electrico', solve_for='prop'), 'base_95': FarmacoMatematica.coulter_2014_iso_cx(0.0, 0.95, efecto='Electrico', solve_for='prop'), 'unit': 'ug/ml'}
                isobolas_tmp['Propofol Tolerancia a la Nocicepcion Visceral (Shen 2026)'] = {'ce_50': FarmacoMatematica.shen_2026_iso_cx(c_y, 0.50, solve_for='prop'), 'ce_95': FarmacoMatematica.shen_2026_iso_cx(c_y, 0.95, solve_for='prop'), 'base_50': 3.84, 'base_95': 5.20, 'unit': 'ug/ml'}

        for k, v in isobolas_tmp.items():
            isobolas_clasicas[k] = {'ce_50': clean_arr(v['ce_50']), 'ce_95': clean_arr(v['ce_95'])}
            
        # 2. MODO DINÁMICO (Evaluado sobre el arreglo Ce real en el tiempo)
        if datos.isobolograma_2d not in ["Ninguno", "Ninguna"] and datos.isobolograma_2d in isobolas_tmp and datos.tipo_grafico_iso != "Clásico":
            req_dict = isobolas_tmp[datos.isobolograma_2d]
            ce_dinamico = Ce
            
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
                elif 'Coulter 2014' in datos.isobolograma_2d:
                    ef_str = 'Electrico' if 'Electrico' in datos.isobolograma_2d else 'CHWSS'
                    req50_arr = FarmacoMatematica.coulter_2014_iso_cx(ce_dinamico, 0.50, ef_str, 'prop')
                    req95_arr = FarmacoMatematica.coulter_2014_iso_cx(ce_dinamico, 0.95, ef_str, 'prop')
                elif 'Shen 2026' in datos.isobolograma_2d:
                    req50_arr = FarmacoMatematica.shen_2026_iso_cx(ce_dinamico, 0.50, 'prop')
                    req95_arr = FarmacoMatematica.shen_2026_iso_cx(ce_dinamico, 0.95, 'prop')

            curvas = []
            nombre_droga = datos.isobolograma_2d.split(' ')[0]
            unit = req_dict['unit']

            if 'Requerimiento' in datos.tipo_grafico_iso:
                curvas.append({"nombre": f"Req. 50% ({unit})", "color": "#ba68c8", "es_linea_punteada": True, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(req50_arr)})
                curvas.append({"nombre": f"Req. 95% ({unit})", "color": "#7b1fa2", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(req95_arr)})
                
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
                curvas.append({"nombre": "Ahorro 50% (%)", "color": "#ba68c8", "es_linea_punteada": True, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(ahorro_50)})
                curvas.append({"nombre": "Ahorro 95% (%)", "color": "#7b1fa2", "es_linea_punteada": False, "es_eje_secundario": False, "es_area": False, "datos": clean_arr(ahorro_95)})

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
                "etiqueta_y2": "EEG Proyectado" if 'Requerimiento' in datos.tipo_grafico_iso else None,
                "limite_y": 105.0 if 'Requerimiento' not in datos.tipo_grafico_iso else None,
                "curvas": curvas
            }

        # 3. SUPERFICIE 3D UNIFICADA
        if datos.modelo_3d not in ['Ninguna', 'Ninguno'] and datos.modelo_3d in isobolas_tmp:
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
            else:
                if 'Jonkman' in datos.modelo_3d:
                    y_max, x_max = max(base_max, 1000.0), 10.0
                    ylab, xlab, zlab = 'Ce Ketamina (ng/ml)', 'Ce Remifentanilo (ng/ml)', datos.modelo_3d
                elif 'Hui' in datos.modelo_3d or 'Shen' in datos.modelo_3d or 'Coulter' in datos.modelo_3d:
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
                    Z = FarmacoMatematica.jonkman_2018_prob(Y, X)
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
                    Z = FarmacoMatematica.jonkman_2018_prob(X, Y)
                elif 'Hui' in datos.modelo_3d:
                    Z = FarmacoMatematica.hui_1995_prob(X, Y, efecto='Tolerancia al Estímulo Eléctrico' if 'Eléctrico' in datos.modelo_3d else 'LOC', as_percent=True)
                elif 'Shen' in datos.modelo_3d:
                    Z = FarmacoMatematica.shen_2026_prob(X, Y, as_percent=True)
                elif 'Coulter' in datos.modelo_3d:
                    Z = FarmacoMatematica.coulter_2014_prob(X, Y, efecto='Electrico' if 'Electrico' in datos.modelo_3d else 'CHWSS', as_percent=True)
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
