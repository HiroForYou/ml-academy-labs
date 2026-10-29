"""Corpus sintético de políticas internas de dos tenants y preguntas etiquetadas.

Synthetic internal-policy corpus for two tenants, plus labelled questions.
Both tenants share the same document templates with different values, so a retriever that ignores
the tenant can return a chunk that looks right but belongs to the other company.
"""

import json
import re
from pathlib import Path

TENANTS = {
    "andina": {
        "company": "Andina Foods",
        "vac_days": 22,
        "vac_notice": 15,
        "vac_carry": 5,
        "meal_limit": 40,
        "receipt_days": 30,
        "approval_limit": 500,
        "pwd_len": 14,
        "lock_min": 5,
        "phish_channel": "abuse@andina.example",
        "remote_days": 3,
        "stipend": 300,
        "oncall_days": 7,
        "sev1_min": 15,
        "log_days": 90,
        "cust_months": 24,
    },
    "boreal": {
        "company": "Boreal Health",
        "vac_days": 18,
        "vac_notice": 30,
        "vac_carry": 10,
        "meal_limit": 55,
        "receipt_days": 45,
        "approval_limit": 250,
        "pwd_len": 16,
        "lock_min": 10,
        "phish_channel": "soc@boreal.example",
        "remote_days": 2,
        "stipend": 450,
        "oncall_days": 14,
        "sev1_min": 10,
        "log_days": 365,
        "cust_months": 36,
    },
}

# Plantillas de documentos: un párrafo por elemento. Document templates: one paragraph per item.
DOCS = {
    "es": {
        "vacaciones": [
            "En {company}, cada persona con contrato indefinido tiene {vac_days} días hábiles de vacaciones pagadas por año "
            "calendario. Los días se acumulan de forma proporcional desde el primer mes de trabajo y se registran en el portal "
            "de recursos humanos.",
            "Las vacaciones se solicitan en el portal con al menos {vac_notice} días de anticipación. La persona responsable "
            "del equipo aprueba o rechaza la solicitud en un plazo de tres días hábiles. Los cierres contables de fin de mes "
            "pueden tener restricciones.",
            "Al terminar el año se pueden trasladar como máximo {vac_carry} días no usados al año siguiente. Los días "
            "trasladados vencen el 31 de marzo. Los días no utilizados no se pagan en efectivo, salvo cuando la ley lo exige "
            "al finalizar el contrato.",
        ],
        "gastos": [
            "En {company}, los gastos de viaje de trabajo se reembolsan con comprobante. El límite para comidas es de "
            "{meal_limit} USD por día de viaje, incluidas las propinas. Las bebidas alcohólicas no se reembolsan.",
            "Los comprobantes se cargan en la aplicación de gastos dentro de los {receipt_days} días siguientes al gasto. "
            "Pasado ese plazo, el reembolso requiere una excepción firmada por el área de finanzas.",
            "Todo gasto individual superior a {approval_limit} USD necesita la aprobación previa del responsable directo. "
            "Los vuelos se reservan en clase económica y los hoteles, con la agencia contratada por la empresa.",
        ],
        "seguridad": [
            "En {company}, las contraseñas de las cuentas corporativas deben tener al menos {pwd_len} caracteres y no pueden "
            "reutilizarse. La autenticación multifactor (MFA) es obligatoria en el correo, la VPN y las aplicaciones internas.",
            "Los portátiles de la empresa usan cifrado de disco completo y se bloquean tras {lock_min} minutos de "
            "inactividad. Está prohibido instalar software que no figure en el catálogo aprobado por TI.",
            "Si recibes un correo sospechoso de phishing, no abras los enlaces y repórtalo a {phish_channel}. El equipo de "
            "seguridad responde en horario laboral y bloquea el remitente cuando confirma el ataque.",
        ],
        "remoto": [
            "En {company}, el personal elegible puede trabajar en remoto hasta {remote_days} días por semana, acordados con "
            "su equipo. Los días presenciales se dedican a reuniones de planificación y trabajo colaborativo.",
            "Para equipar el espacio de trabajo en casa, la empresa entrega un pago único de {stipend} USD. Ese monto cubre "
            "silla, monitor o accesorios, y se solicita con la factura en la aplicación de gastos.",
            "Trabajar desde otro país requiere autorización previa de recursos humanos y del área legal, por sus "
            "implicaciones fiscales. Las conexiones desde redes públicas deben pasar siempre por la VPN corporativa.",
        ],
        "guardias": [
            "En {company}, los equipos de plataforma mantienen una rotación de guardia de {oncall_days} días. La persona de "
            "guardia recibe las alertas en el teléfono y debe tener acceso a un portátil durante todo el turno.",
            "Ante un incidente de severidad 1, la persona de guardia confirma la alerta en un máximo de {sev1_min} minutos y "
            "abre un canal de crisis. Los incidentes de severidad 2 se atienden en horario laboral.",
            "Después de cada incidente de severidad 1 se redacta un análisis posterior sin culpables en cinco días hábiles. "
            "El análisis incluye la cronología, la causa raíz y las acciones preventivas.",
        ],
        "retencion": [
            "En {company}, los registros (logs) de las aplicaciones se conservan durante {log_days} días y después se "
            "eliminan de forma automática. Los registros de auditoría de acceso se guardan aparte, en almacenamiento "
            "inmutable.",
            "Los datos personales de clientes se eliminan {cust_months} meses después del cierre de la cuenta, salvo "
            "obligación legal de conservarlos. Las solicitudes de borrado anticipado se atienden en menos de treinta días.",
            "Las copias de seguridad se cifran y se replican en una segunda región. Cada trimestre se prueba la restauración "
            "de una copia y el resultado se documenta en el plan de continuidad del negocio.",
        ],
    },
    "en": {
        "vacation": [
            "At {company}, every permanent employee gets {vac_days} business days of paid vacation per calendar year. Days "
            "accrue proportionally from the first month of work and are recorded in the human resources portal.",
            "Vacation is requested in the portal at least {vac_notice} days in advance. The team lead approves or rejects "
            "the request within three business days. Month-end accounting closes may have restrictions.",
            "At year end, a maximum of {vac_carry} unused days can be carried over to the next year. Carried-over days "
            "expire on March 31. Unused days are not paid out in cash, except when the law requires it at contract end.",
        ],
        "expenses": [
            "At {company}, business travel expenses are reimbursed with a receipt. The meal limit is {meal_limit} USD per "
            "travel day, tips included. Alcoholic drinks are not reimbursed.",
            "Receipts are uploaded to the expense app within {receipt_days} days after the expense. After that deadline, "
            "reimbursement requires an exception signed by the finance team.",
            "Any single expense above {approval_limit} USD needs prior approval from the direct manager. Flights are booked "
            "in economy class and hotels through the company's contracted agency.",
        ],
        "security": [
            "At {company}, corporate account passwords must be at least {pwd_len} characters long and cannot be reused. "
            "Multi-factor authentication (MFA) is mandatory for email, the VPN, and internal applications.",
            "Company laptops use full-disk encryption and lock after {lock_min} minutes of inactivity. Installing software "
            "that is not in the IT-approved catalog is forbidden.",
            "If you receive a suspicious phishing email, do not open the links and report it to {phish_channel}. The "
            "security team responds during business hours and blocks the sender once the attack is confirmed.",
        ],
        "remote": [
            "At {company}, eligible staff can work remotely up to {remote_days} days per week, agreed with their team. "
            "Office days are used for planning meetings and collaborative work.",
            "To equip the home workspace, the company provides a one-time payment of {stipend} USD. It covers a chair, a "
            "monitor, or accessories, and is requested with the invoice in the expense app.",
            "Working from another country requires prior approval from human resources and legal because of tax "
            "implications. Connections from public networks must always go through the corporate VPN.",
        ],
        "oncall": [
            "At {company}, platform teams keep an on-call rotation of {oncall_days} days. The on-call engineer receives "
            "alerts on the phone and must have access to a laptop for the whole shift.",
            "For a severity 1 incident, the on-call engineer acknowledges the alert within {sev1_min} minutes at most and "
            "opens a crisis channel. Severity 2 incidents are handled during business hours.",
            "After every severity 1 incident, a blameless post-incident review is written within five business days. The "
            "review includes the timeline, the root cause, and preventive actions.",
        ],
        "retention": [
            "At {company}, application logs are kept for {log_days} days and then deleted automatically. Access audit logs "
            "are stored separately, in immutable storage.",
            "Customer personal data is deleted {cust_months} months after the account is closed, unless the law requires "
            "keeping it. Early deletion requests are handled in less than thirty days.",
            "Backups are encrypted and replicated to a second region. A restore is tested every quarter, and the result is "
            "documented in the business continuity plan.",
        ],
    },
}

# Hechos preguntables: documento (índice en DOCS), respuesta esperada y pregunta. Las preguntas no nombran la empresa.
# Answerable facts: document (index into DOCS), expected answer, and question. Questions don't name the company.
FACTS = [
    ("q01", 0, "{vac_days} días hábiles", "{vac_days} business days",
     "¿Cuántos días de vacaciones pagadas tengo al año?", "How many days of paid vacation do I get per year?"),
    ("q02", 0, "{vac_notice} días de anticipación", "{vac_notice} days in advance",
     "¿Con cuánta anticipación debo pedir mis vacaciones?", "How far in advance must I request vacation?"),
    ("q03", 0, "{vac_carry} días no usados", "{vac_carry} unused days",
     "¿Cuántos días libres que no tomé puedo pasar al próximo año?", "How many days off I didn't take can I move to next year?"),
    ("q04", 1, "{meal_limit} USD por día", "{meal_limit} USD per travel day",
     "¿Cuál es el tope diario para comer durante un viaje?", "What is the daily cap for food on a trip?"),
    ("q05", 1, "{receipt_days} días siguientes", "within {receipt_days} days",
     "¿Qué plazo tengo para subir los comprobantes de un gasto?", "What is the deadline to upload expense receipts?"),
    ("q06", 1, "superior a {approval_limit} USD", "above {approval_limit} USD",
     "¿A partir de qué monto un gasto necesita aprobación previa?", "From what amount does an expense need prior approval?"),
    ("q07", 2, "{pwd_len} caracteres", "{pwd_len} characters",
     "¿Qué longitud mínima debe tener la contraseña?", "What is the minimum password length?"),
    ("q08", 2, "{lock_min} minutos de inactividad", "{lock_min} minutes of inactivity",
     "¿Tras cuánto tiempo sin uso se bloquea el portátil?", "After how long without use does the laptop lock?"),
    ("q09", 2, "{phish_channel}", "{phish_channel}",
     "¿Dónde reporto un correo de phishing?", "Where do I report a phishing email?"),
    ("q10", 3, "{remote_days} días por semana", "{remote_days} days per week",
     "¿Cuántos días a la semana puedo teletrabajar?", "How many days a week can I work from home?"),
    ("q11", 3, "pago único de {stipend} USD", "payment of {stipend} USD",
     "¿Cuánto dinero dan para equipar la oficina en casa?", "How much money do I get to set up my home office?"),
    ("q12", 4, "guardia de {oncall_days} días", "rotation of {oncall_days} days",
     "¿Cuánto dura cada rotación de guardia?", "How long is each on-call rotation?"),
    ("q13", 4, "{sev1_min} minutos", "{sev1_min} minutes",
     "¿En cuántos minutos hay que confirmar una alerta de severidad 1?",
     "Within how many minutes must a severity 1 alert be acknowledged?"),
    ("q14", 5, "{log_days} días", "{log_days} days",
     "¿Cuánto tiempo se guardan los logs de las aplicaciones?", "How long are application logs kept?"),
    ("q15", 5, "{cust_months} meses", "{cust_months} months",
     "¿Cuándo se borran los datos personales de un cliente que cerró su cuenta?",
     "When is a customer's personal data deleted after they close their account?"),
]  # fmt: skip


def documents(lang: str = "es") -> list[dict]:
    """Un documento por tenant y plantilla, con metadatos. One document per tenant and template, with metadata."""
    docs = []
    for tenant, values in TENANTS.items():
        for name, paragraphs in DOCS[lang].items():
            docs.append(
                {
                    "doc_id": f"{tenant}/{name}",
                    "tenant": tenant,
                    "topic": name,
                    "text": "\n\n".join(p.format(**values) for p in paragraphs),
                }
            )
    return docs


def queries(lang: str = "es") -> list[dict]:
    """Una pregunta por hecho y tenant, con la evidencia (span de la frase) en su documento.

    One question per fact and tenant, with the evidence (sentence span) inside its document.
    """
    by_id = {d["doc_id"]: d for d in documents(lang)}
    topics = list(DOCS[lang])
    out = []
    for qid, doc_idx, ans_es, ans_en, q_es, q_en in FACTS:
        answer_tpl, question = (ans_es, q_es) if lang == "es" else (ans_en, q_en)
        for tenant, values in TENANTS.items():
            doc = by_id[f"{tenant}/{topics[doc_idx]}"]
            answer = answer_tpl.format(**values)
            pos = doc["text"].find(answer)
            assert pos >= 0 and doc["text"].count(answer) == 1, (qid, tenant, answer)
            start, end = _sentence_span(doc["text"], pos)
            out.append(
                {
                    "query_id": f"{tenant}-{qid}",
                    "tenant": tenant,
                    "question": question,
                    "answer": answer,
                    "doc_id": doc["doc_id"],
                    "evidence": (start, end),
                }
            )
    return out


def _sentence_span(text: str, pos: int) -> tuple[int, int]:
    """Límites de la frase que contiene pos. Bounds of the sentence that contains pos."""
    starts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text[:pos])]
    end = re.search(r"[.!?](\s|$)", text[pos:])
    return starts[-1], pos + end.start() + 1 if end else len(text)


def export(docs: list[dict], folder: Path) -> None:
    """Guarda cada documento con su archivo de metadatos, como en un origen S3 de Bedrock Knowledge Bases.

    Saves each document with its metadata file, as in an S3 data source for Bedrock Knowledge Bases.
    """
    for doc in docs:
        path = folder / f"{doc['doc_id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc["text"], encoding="utf-8")
        meta = {"metadataAttributes": {"tenant": doc["tenant"], "topic": doc["topic"]}}
        Path(f"{path}.metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
