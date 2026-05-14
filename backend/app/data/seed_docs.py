from pathlib import Path

DOCS = {
    "it_support_runbook.md": """# IT Support Runbook

## Access and VPN incidents
If a user cannot access VPN after a password reset, first verify inventory for available VPN licenses. If no license is available, submit a procurement request for VPN licenses and create a support ticket with severity high when the user is blocked from customer or production work.

## Laptop replacement
Laptop replacement is approved when the device blocks job critical work, has repeated failures, or is needed for a new employee. Check laptop inventory first. If available stock is below the reorder threshold, submit procurement before confirming the replacement path. Engineering onboarding laptops should target a maximum budget of 2200 USD.

## Escalation quality bar
Every support escalation should include user impact, system affected, urgency, and the action already attempted by the agent.
""",
    "compliance_policy.md": """# Compliance and Data Handling Policy

## Customer data
Customer personal data must not be exported into personal email, public notebooks, or unmanaged AI tools. If a request involves moving customer data outside approved systems, the agent must flag a policy risk and explain the safe alternative.

## AI usage
Internal AI tools may summarize approved internal documents. They must not invent policy, expose secrets, or send restricted customer records to third party APIs. For this assessment the agent runs locally, so document lookup and tool execution stay on the user's machine.

## Evidence standard
A policy flag must include the policy area, concern, and evidence that triggered the risk. The flag is not a punishment. It creates a review trail for the operations team.
""",
    "procurement_rules.md": """# Procurement Rules

## Standard approvals
Requests under 500 USD can be submitted directly when tied to business need. Requests from 500 to 2500 USD require manager review. Requests above 2500 USD require finance review.

## Reorder workflow
When inventory is below threshold, the agent should submit a procurement request with the quantity needed to restore healthy stock. Hardware requests should include expected team impact and maximum budget.

## Preferred quantities
VPN licenses should be ordered in batches of 10. GitHub Enterprise seats should be ordered in batches of 20 when available stock is below threshold. Laptops should be ordered in batches of 5 when the stock is below threshold.
""",
    "budget_context.md": """# Q2 Operations Budget Context

The platform operations budget has 7800 USD unallocated for urgent tooling, hardware replacement, and access management. The recommended priority order is production access blockers first, employee onboarding blockers second, and general productivity improvements third.

For local demos, budget checks are advisory. The agent should still show the estimated action and explain when a manager or finance review is likely required.
""",
}


def seed() -> None:
    base = Path(__file__).resolve().parents[2] / "data" / "docs"
    base.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCS.items():
        (base / filename).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    seed()
    print("Seed documents written.")
