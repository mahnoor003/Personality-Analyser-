from fpdf import FPDF

def generate_report(name, traits, source="LinkedIn"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    title = f"{source} Personality Report for {name}"
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)

    for trait, score in traits.items():
        pdf.cell(200, 10, txt=f"{trait}: {score:.2f}", ln=True)

    safe_name = name.replace(' ', '_').replace('/', '_')
    filename = f"{source.lower()}_report_{safe_name}.pdf"
    pdf.output(filename)
    return filename
