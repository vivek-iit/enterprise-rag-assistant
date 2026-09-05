"""Generate sample policy documents for the Enterprise RAG Assistant demo.

Produces three files inside ``data/sample_docs/``:
- Leave_Policy.pdf       – annual leave, sick leave, carry-forward rules
- IT_Security_Policy.txt – password policy, VPN, clean desk, incident contacts
- Employee_Benefits.docx – health insurance, wellness allowance, education reimbursement
"""

from pathlib import Path

from fpdf import FPDF
import docx

OUTPUT_DIR = Path("data/sample_docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _heading(pdf: FPDF, text: str, size: int = 13) -> None:
    pdf.set_font("Helvetica", "B", size)
    pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)


def _body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)
    pdf.ln(3)


# ── 1. Leave_Policy.pdf ───────────────────────────────────────────────────────

def create_leave_policy_pdf() -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Leave Policy", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Effective Date: 01 January 2024  |  Owner: Human Resources Department",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    _heading(pdf, "1. Annual Leave", 13)
    _body(pdf,
          "All permanent employees are entitled to 20 days of paid annual leave per calendar year. "
          "Leave accrues at 1.67 days per month of service and may not be taken during the probation "
          "period without prior written approval from HR. Requests must be submitted at least 5 working "
          "days in advance via the HR portal.")

    _heading(pdf, "2. Sick Leave", 13)
    _body(pdf,
          "Employees are entitled to 10 days of paid sick leave per year. Absence exceeding 2 consecutive "
          "days requires a medical certificate from a registered practitioner. Sick leave does not carry "
          "forward to the following year and is not encashable at any point.")

    _heading(pdf, "3. Carry-Forward Rules", 13)
    _body(pdf,
          "Unused annual leave may be carried forward to the following calendar year subject to a "
          "maximum of 5 days per year. Carry-forward leave must be utilized within Q1 "
          "(by 31 March) of the subsequent year. Any balance remaining after 31 March lapses "
          "without compensation. Employees wishing to encash leave beyond the carry-forward cap must "
          "apply to HR no later than 31 December of the current year.")

    _heading(pdf, "4. Other Leave Types", 13)
    _body(pdf,
          "Maternity Leave: 26 weeks paid leave for female employees after 80 days of continuous service.\n"
          "Paternity Leave: 5 days paid leave to be taken within 6 months of the child's birth.\n"
          "Bereavement Leave: 3 paid days for the death of an immediate family member.\n"
          "Study / Examination Leave: Up to 5 days per year for approved professional examinations.")

    _heading(pdf, "5. General Conditions", 13)
    _body(pdf,
          "Leave balances are visible in real time on the employee self-service portal. "
          "Leave taken in excess of entitlement will be treated as unpaid leave. "
          "This policy is subject to annual review and may be amended at the discretion of the "
          "Board of Directors in compliance with applicable labour legislation.")

    pdf.output(str(OUTPUT_DIR / "Leave_Policy.pdf"))
    print("[OK] Created Leave_Policy.pdf")


# ── 2. IT_Security_Policy.txt ─────────────────────────────────────────────────

def create_it_security_policy_txt() -> None:
    content = """\
IT SECURITY POLICY
==================
Effective Date: 01 January 2024
Document Owner: Chief Information Security Officer (CISO)
Classification: Internal Use Only
Review Cycle: Annual

1. PASSWORD POLICY
------------------
All employees must adhere to the following password guidelines:
- Passwords must be at least 12 characters long.
- Passwords must contain uppercase letters, lowercase letters, digits, and special characters (!@#$%^&*).
- Passwords must not include the employee's name, username, or date of birth.
- Passwords expire every 90 days; a reminder is sent 10 days before expiry.
- The last 10 passwords may not be reused.
- Multi-Factor Authentication (MFA) is mandatory for all corporate systems and email accounts.
- Passwords must never be shared. Report any suspected credential compromise immediately.
- Use of the company-approved password manager (LastPass Enterprise) is strongly encouraged.

2. VPN USAGE POLICY
-------------------
- All remote access to internal systems must be conducted exclusively via the corporate VPN.
- Employees must connect to vpn.company.internal before accessing any internal resource outside the office.
- Split-tunnelling is disabled; all traffic routes through the corporate VPN when connected.
- Personal devices must have the latest OS security patches before connecting to the corporate VPN.
- VPN credentials are personal and non-transferable. Suspected compromise must be reported within 1 hour.
- VPN connection logs are retained for 12 months and audited quarterly by the IT Security team.

3. CLEAN DESK POLICY
--------------------
- No confidential documents or sensitive materials may be left visible on desks when away from the workstation.
- Screens must be locked (Windows: Win+L; macOS: Cmd+Ctrl+Q) whenever leaving the desk, even briefly.
- Printed confidential documents must be collected from the printer immediately and stored securely.
- Sensitive printed materials must be disposed of using the cross-cut shredders on each floor.
- Portable storage devices (USB drives, external hard disks) are prohibited unless approved by IT Security.
- Whiteboards containing confidential information must be fully erased at the end of each working day.
- Visitor access badges must be visibly worn at all times within the office premises.

4. INCIDENT RESPONSE CONTACTS
------------------------------
Report any suspected security incident immediately to the following contacts:

Primary Contact:
  IT Security Helpdesk
  Email : security@company.internal
  Phone : +1-800-SEC-HELP (800-732-4357)
  Hours : 24 x 7 x 365

Escalation Contact:
  Chief Information Security Officer (CISO)
  Name  : Jane Doe
  Email : ciso@company.internal
  Phone : +1-800-555-0199

Data Breach / Privacy Concerns:
  Data Protection Officer (DPO)
  Email : dpo@company.internal
  Phone : +1-800-555-0177

Emergency (Active Threat / Ransomware):
  Internal Extension : 9911
  External Line      : +1-800-555-9911
  The Incident Response Team (IRT) will be activated within 15 minutes of notification.

5. ACCEPTABLE USE
-----------------
- Company IT assets (laptops, mobiles, email) are primarily for business use.
- Limited personal use is permitted provided it does not impact productivity or violate other policies.
- The following are strictly prohibited: accessing illegal content, torrenting, cryptocurrency mining,
  and installing unauthorized software on company devices.
- All company-owned devices are subject to monitoring and periodic security audits.
- Violation of this policy may result in disciplinary action, up to and including termination.
"""
    (OUTPUT_DIR / "IT_Security_Policy.txt").write_text(content, encoding="utf-8")
    print("[OK] Created IT_Security_Policy.txt")


# ── 3. Employee_Benefits.docx ─────────────────────────────────────────────────

def create_employee_benefits_docx() -> None:
    doc = docx.Document()

    doc.add_heading("Employee Benefits Guide", level=0)
    doc.add_paragraph("Effective Date: 01 January 2024  |  Human Resources Department")
    doc.add_paragraph(
        "This guide summarises the benefits available to all full-time employees. "
        "Contact hr@company.internal for queries or to initiate a claim."
    )

    # ── Health Insurance ──────────────────────────────────────────────────────
    doc.add_heading("1. Health Insurance", level=1)
    doc.add_paragraph(
        "The company provides comprehensive group health insurance for all full-time employees "
        "and their immediate dependents (spouse and up to two children under 21 years of age)."
    )
    p = doc.add_paragraph()
    p.add_run("Coverage Details:\n").bold = True
    p.add_run(
        "- In-patient hospitalization: up to $200,000 per annum per insured family.\n"
        "- Out-patient consultations: up to $2,000 per annum per employee.\n"
        "- Dental coverage: up to $500 per annum per employee.\n"
        "- Vision care (frames + lenses or contact lenses): up to $300 per annum.\n"
        "- Maternity benefits (pre- and post-natal): up to $5,000 per delivery.\n"
        "- Pre-existing conditions: covered after a 12-month waiting period.\n"
        "- Emergency overseas medical evacuation: up to $500,000."
    )
    doc.add_paragraph(
        "Enrolment must be completed within 30 days of joining. "
        "Changes to dependent coverage are only permitted during the annual open-enrolment window "
        "(November) or within 30 days of a qualifying life event (marriage, birth, adoption)."
    )

    # ── Wellness Allowance ────────────────────────────────────────────────────
    doc.add_heading("2. Wellness Allowance", level=1)
    doc.add_paragraph(
        "Each employee receives an annual Wellness Allowance of $500 to support physical and "
        "mental well-being. The allowance is credited to the benefits wallet on 1 January each year "
        "(pro-rated for employees who join mid-year)."
    )
    p = doc.add_paragraph()
    p.add_run("Eligible Expenses:\n").bold = True
    p.add_run(
        "- Gym memberships and fitness classes (yoga, pilates, CrossFit, swimming, etc.).\n"
        "- Sports equipment purchases up to $200 per item.\n"
        "- Mental health and counselling sessions with approved providers.\n"
        "- Nutrition and dietitian consultations.\n"
        "- Mindfulness app subscriptions (Calm, Headspace, etc.).\n"
        "- Health screening packages not covered under group insurance."
    )
    doc.add_paragraph(
        "Claims must be submitted through the HR portal with valid receipts within 60 days of the "
        "expense date. Unused wellness allowance does not carry forward to the following year."
    )

    # ── Education Reimbursement ───────────────────────────────────────────────
    doc.add_heading("3. Education Reimbursement", level=1)
    doc.add_paragraph(
        "The company supports continuous learning through an Education Reimbursement programme "
        "for job-related courses, certifications, and degree programmes."
    )
    p = doc.add_paragraph()
    p.add_run("Reimbursement Limits:\n").bold = True
    p.add_run(
        "- Professional certifications (AWS, PMP, CPA, CFA, etc.): up to $2,000 per certification.\n"
        "- Online courses and MOOCs (Coursera, Udemy, LinkedIn Learning): up to $500 per year.\n"
        "- Undergraduate or postgraduate degree programmes: up to $5,000 per academic year.\n"
        "- Conference attendance (registration fees only): up to $1,500 per event."
    )
    p = doc.add_paragraph()
    p.add_run("Eligibility Criteria:\n").bold = True
    p.add_run(
        "- At least 6 months of continuous service required.\n"
        "- The course or programme must be relevant to the employee's current role or career growth.\n"
        "- Pre-approval from the line manager and HR is mandatory before enrolment.\n"
        "- A passing grade (or equivalent completion certificate) is required for reimbursement.\n"
        "- Employees who resign within 12 months of receiving reimbursement must repay 100% of the amount."
    )

    # ── Additional Benefits ───────────────────────────────────────────────────
    doc.add_heading("4. Additional Benefits", level=1)
    extras = [
        ("Retirement Plan (401k)",
         "Company matches employee 401(k) contributions dollar-for-dollar up to 5% of base salary."),
        ("Group Life Insurance",
         "Term life insurance of 3× annual base salary provided at no cost to the employee."),
        ("Flexible / Hybrid Work",
         "Hybrid work model: minimum 3 days in-office and up to 2 days remote per week."),
        ("Parental Leave",
         "16 weeks paid primary caregiver leave; 4 weeks paid secondary caregiver leave."),
        ("Employee Assistance Programme (EAP)",
         "Free confidential counselling via LifeWorks — up to 6 sessions per year per employee."),
        ("Commuter Benefits",
         "Pre-tax commuter benefit of up to $300/month for public transit or qualified parking."),
    ]
    for title, detail in extras:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{title}: ").bold = True
        p.add_run(detail)

    doc.add_heading("5. Policy Review", level=1)
    doc.add_paragraph(
        "This Benefits Guide is reviewed annually by the HR department and is subject to change. "
        "Employees will be notified of material changes at least 30 days in advance. "
        "For the most current version, refer to the HR portal or contact hr@company.internal."
    )

    doc.save(str(OUTPUT_DIR / "Employee_Benefits.docx"))
    print("[OK] Created Employee_Benefits.docx")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Writing sample documents to: {OUTPUT_DIR.resolve()}\n")
    create_leave_policy_pdf()
    create_it_security_policy_txt()
    create_employee_benefits_docx()
    print(f"\nDone. Files in {OUTPUT_DIR.resolve()}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
