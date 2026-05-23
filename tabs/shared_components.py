"""
tabs/shared_components.py — Shared UI utilities
─────────────────────────────────────────────────
Reusable components extracted from tab files so they can be imported
into any tab without duplication.

Currently extracted from:
  - tab5_similar_patients  (_kpi_card)
  - tab6_cohort_monitoring (_kpi_card — identical signature)
  - tab8_model_performance (_kpi_card — formerly used "border" param name)
"""


def kpi_card(label: str, value: str, sub: str = "", color: str = "#2b6cb0") -> str:
    """
    Returns an HTML string for a KPI metric card.

    Args:
        label: Short uppercase label shown below the value.
        value: Primary metric value (string, e.g. "74%").
        sub:   Optional smaller subtitle line beneath the label.
        color: Accent colour for the top border stripe.
    """
    sub_html = (
        f"<div style='font-size:0.71rem;color:#a0aec0;margin-top:2px;'>{sub}</div>"
        if sub else ""
    )
    return f"""
    <div style="background:white;border-radius:10px;padding:1.1rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;
                border-top:3px solid {color};">
      <div style="font-size:1.7rem;font-weight:700;color:#1a365d;line-height:1.2;">{value}</div>
      <div style="font-size:0.73rem;font-weight:600;color:#718096;
                  text-transform:uppercase;letter-spacing:0.05em;margin-top:3px;">{label}</div>
      {sub_html}
    </div>"""
