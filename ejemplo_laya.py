"""Ejemplo minimo de Laya: triaje de un ticket de soporte."""
import time

from laya import Router

router = Router()  # descarga el checkpoint la primera vez

state = (
    "Hi, we were billed twice for March. Please refund the duplicate today "
    "or we will cancel our plan."
)

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

t0 = time.perf_counter()
result = router.predict(state, questions)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"modelo enrutado : {result['routing']['model']}")
print(f"latencia        : {elapsed_ms:.1f} ms")
print(f"department      : {result['answers']['department']['choice']}")
print(f"urgency         : {result['answers']['urgency']['score']}")
print(f"churn_risk      : {result['answers']['churn_risk']['noul']:.3f}")

print("\n--- multilingue ---")
for text in [
    "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
    "La aplicación se cierra cada vez que abro la configuración.",
]:
    r = router.predict(text, {"department": questions["department"]})
    print(r["routing"]["model"], "->", r["answers"]["department"]["choice"])
